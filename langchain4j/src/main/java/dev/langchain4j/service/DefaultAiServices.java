package dev.langchain4j.service;

import static dev.langchain4j.internal.Exceptions.illegalArgument;
import static dev.langchain4j.internal.Utils.isNullOrEmpty;
import static dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA;
import static dev.langchain4j.model.chat.request.ResponseFormatType.JSON;
import static dev.langchain4j.model.output.FinishReason.TOOL_EXECUTION;
import static dev.langchain4j.service.AiServiceParamsUtil.chatRequestParameters;
import static dev.langchain4j.service.AiServiceParamsUtil.findArgumentOfType;
import static dev.langchain4j.service.AiServiceValidation.validateParameters;
import static dev.langchain4j.service.IllegalConfigurationException.illegalConfiguration;
import static dev.langchain4j.service.TypeUtils.getRawClass;
import static dev.langchain4j.service.TypeUtils.isImageType;
import static dev.langchain4j.service.TypeUtils.resolveFirstGenericParameterClass;
import static dev.langchain4j.service.TypeUtils.typeHasRawClass;
import static dev.langchain4j.spi.ServiceHelper.loadFactories;

import dev.langchain4j.Internal;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.guardrail.ChatExecutor;
import dev.langchain4j.guardrail.GuardrailRequestParams;
import dev.langchain4j.guardrail.InputGuardrailRequest;
import dev.langchain4j.guardrail.OutputGuardrailRequest;
import dev.langchain4j.internal.DefaultExecutorProvider;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.invocation.InvocationParameters;
import dev.langchain4j.invocation.LangChain4jManaged;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.moderation.Moderation;
import dev.langchain4j.observability.api.event.AiServiceCompletedEvent;
import dev.langchain4j.observability.api.event.AiServiceErrorEvent;
import dev.langchain4j.observability.api.event.AiServiceResponseReceivedEvent;
import dev.langchain4j.observability.api.event.AiServiceStartedEvent;
import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.rag.query.Metadata;
import dev.langchain4j.service.guardrail.GuardrailService;
import dev.langchain4j.service.memory.ChatMemoryAccess;
import dev.langchain4j.service.memory.ChatMemoryService;
import dev.langchain4j.service.output.ServiceOutputParser;
import dev.langchain4j.service.tool.ToolServiceContext;
import dev.langchain4j.service.tool.ToolServiceResult;
import dev.langchain4j.spi.services.TokenStreamAdapter;
import java.io.InputStream;
import java.lang.annotation.Annotation;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.lang.reflect.Proxy;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Scanner;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

@Internal
class DefaultAiServices<T> extends AiServices<T> {

    private final ServiceOutputParser serviceOutputParser = new ServiceOutputParser();
    private final Collection<TokenStreamAdapter> tokenStreamAdapters = loadFactories(TokenStreamAdapter.class);

    private static final Set<Class<? extends Annotation>> VALID_PARAM_ANNOTATIONS =
            Set.of(dev.langchain4j.service.UserMessage.class, V.class, MemoryId.class, UserName.class);

    DefaultAiServices(AiServiceContext context) {
        super(context);
    }

    protected void validate() {
        performBasicValidation();
        AiServiceValidation.validate(context);
    }

    private Object handleChatMemoryAccess(Method method, Object[] args) {
        return switch (method.getName()) {
            case "getChatMemory" -> context.chatMemoryService.getChatMemory(args[0]);
            case "evictChatMemory" -> context.chatMemoryService.evictChatMemory(args[0]) != null;
            default ->
                throw new UnsupportedOperationException(
                        "Unknown method on ChatMemoryAccess class : " + method.getName());
        };
    }

    /**
     * 构建 AI Service 的动态代理对象。
     *
     * @return 业务接口的代理实例
     */
    public T build() {
        // 先执行配置与依赖校验，尽早失败，避免请求阶段才暴露配置问题。
        validate();

        // 通过 JDK 动态代理统一拦截接口调用，便于在单一入口完成提示词组装、工具调用与模型分发。
        Object proxyInstance = Proxy.newProxyInstance(
                context.aiServiceClass.getClassLoader(),
                new Class<?>[] {context.aiServiceClass},
                new InvocationHandler() {

                    /**
                     * 拦截业务接口上的每次方法调用，并路由到 AI Service 执行链路。
                     *
                     * @param proxy  代理对象
                     * @param method 当前调用的方法
                     * @param args   方法参数
                     * @return 方法执行结果
                     * @throws Throwable 执行过程中抛出的异常
                     */
                    @Override
                    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                        // 默认方法应沿用接口自身实现，避免被 AI Service 逻辑误处理。
                        if (method.isDefault()) {
                            return InvocationHandler.invokeDefault(proxy, method, args);
                        }

                        // Object 基础方法需单独处理，保证代理对象语义稳定。
                        if (method.getDeclaringClass() == Object.class) {
                            switch (method.getName()) {
                                case "equals":
                                    return proxy == args[0];
                                case "hashCode":
                                    return System.identityHashCode(proxy);
                                case "toString":
                                    return context.aiServiceClass.getName() + "@"
                                            + Integer.toHexString(System.identityHashCode(proxy));
                                default:
                                    throw new IllegalStateException("Unexpected Object method: " + method);
                            }
                        }

                        if (method.getDeclaringClass() == ChatMemoryAccess.class) {
                            // ChatMemoryAccess 属于框架辅助接口，直接走内存访问分支，不进入模型调用链。
                            return handleChatMemoryAccess(method, args);
                        }

                        // TODO 在创建 AI Service 时预校验一次，避免每次调用重复校验。
                        // 在调用前执行参数注解校验，避免非法签名进入后续复杂流程。
                        validateParameters(context.aiServiceClass, method);

                        // 调用上下文在这里统一构建，后续链路依赖它传递追踪信息与内存标识。
                        InvocationParameters invocationParameters = findArgumentOfType(
                                        InvocationParameters.class, args, method.getParameters())
                                .orElseGet(InvocationParameters::new);

                        InvocationContext invocationContext = InvocationContext.builder()
                                .invocationId(UUID.randomUUID())
                                .interfaceName(context.aiServiceClass.getName())
                                .methodName(method.getName())
                                .methodArguments(args != null ? Arrays.asList(args) : List.of())
                                // 内存 ID 为空时回退到默认值，保证无 @MemoryId 的场景也能稳定运行。
                                .chatMemoryId(findMemoryId(method, args).orElse(ChatMemoryService.DEFAULT))
                                .invocationParameters(invocationParameters)
                                .managedParameters(LangChain4jManaged.current())
                                .timestampNow()
                                .build();
                        try {
                            // 进入核心执行流程（提示词构建、模型调用、工具循环与结果解析）。
                            return invoke(method, args, invocationContext);
                        } catch (Exception ex) {
                            // 先发错误事件再抛出，保证可观测性与调用方异常语义同时成立。
                            context.eventListenerRegistrar.fireEvent(AiServiceErrorEvent.builder()
                                    .invocationContext(invocationContext)
                                    .error(ex)
                                    .build());
                            throw ex;
                        }
                    }

                    /**
                     * 执行 AI Service 的核心流程：消息准备、增强、护栏、模型调用与结果转换。
                     *
                     * @param method            当前调用的方法
                     * @param args              方法参数
                     * @param invocationContext 本次调用上下文
                     * @return 方法返回值（流式场景为 {@link TokenStream} 或其适配类型）
                     */
                    public Object invoke(Method method, Object[] args, InvocationContext invocationContext) {

                        // 从上下文中提取内存标识，后续消息拼装与记忆写入都依赖该标识。
                        Object memoryId = invocationContext.chatMemoryId();
                        // 若启用了 chat memory，则按 memoryId 拉取会话记忆；否则保持无状态调用。
                        ChatMemory chatMemory = context.hasChatMemory()
                                ? context.chatMemoryService.getOrCreateChatMemory(memoryId)
                                : null;

                        // 准备 system/user 模板与变量，确保注解模板和运行时参数能正确渲染。
                        Optional<SystemMessage> systemMessage = prepareSystemMessage(memoryId, method, args);
                        var userMessageTemplate = getUserMessageTemplate(memoryId, method, args);
                        var variables = InternalReflectionVariableResolver.findTemplateVariables(
                                userMessageTemplate, method, args);
                        UserMessage originalUserMessage =
                                prepareUserMessage(method, args, userMessageTemplate, variables);

                        // 在真正调用模型前发送“请求开始”事件，便于链路跟踪与审计。
                        context.eventListenerRegistrar.fireEvent(AiServiceStartedEvent.builder()
                                .invocationContext(invocationContext)
                                .systemMessage(systemMessage)
                                .userMessage(originalUserMessage)
                                .build());

                        // 增强前先保留原始用户消息，RAG 场景会替换为增强后的消息内容。
                        UserMessage userMessageForAugmentation = originalUserMessage;

                        AugmentationResult augmentationResult = null;
                        if (context.retrievalAugmentor != null) {
                            // 为 RAG 构造元数据，携带系统消息、历史对话与调用上下文。
                            List<ChatMessage> chatMemoryMessages = chatMemory != null ? chatMemory.messages() : null;
                            Metadata metadata = Metadata.builder()
                                    .chatMessage(userMessageForAugmentation)
                                    .systemMessage(systemMessage.orElse(null))
                                    .chatMemory(chatMemoryMessages)
                                    .invocationContext(invocationContext)
                                    .build();
                            // 执行内容增强并替换当前用户消息，后续模型请求使用增强结果。
                            AugmentationRequest augmentationRequest =
                                    new AugmentationRequest(userMessageForAugmentation, metadata);
                            augmentationResult = context.retrievalAugmentor.augment(augmentationRequest);
                            userMessageForAugmentation = (UserMessage) augmentationResult.chatMessage();
                        }

                        // 汇总护栏执行所需的公共参数，避免输入/输出护栏重复组装上下文。
                        var commonGuardrailParam = GuardrailRequestParams.builder()
                                .chatMemory(chatMemory)
                                .augmentationResult(augmentationResult)
                                .userMessageTemplate(userMessageTemplate)
                                .invocationContext(invocationContext)
                                .aiServiceListenerRegistrar(context.eventListenerRegistrar)
                                .variables(variables)
                                .build();

                        // 先执行输入护栏，再进入模型调用，确保不合规输入不会继续向下游传播。
                        UserMessage userMessage = invokeInputGuardrails(
                                context.guardrailService(), method, userMessageForAugmentation, commonGuardrailParam);

                        // 解析方法返回类型，决定走流式链路还是同步链路。
                        Type returnType = context.returnType != null ? context.returnType : method.getGenericReturnType();
                        boolean streaming = returnType == TokenStream.class || canAdaptTokenStreamTo(returnType);

                        // TODO 评估 returnType==String 时是否也应走该分支。
                        // 检测模型能力后再决定结构化输出策略，避免给不支持能力的模型下发 schema。
                        boolean supportsJsonSchema = supportsJsonSchema();
                        Optional<JsonSchema> jsonSchema = Optional.empty();
                        boolean returnsImage = isImage(returnType);

                        // 非流式且支持 JSON Schema 时，优先根据返回类型推导结构化约束。
                        if (supportsJsonSchema && !streaming && !returnsImage) {
                            jsonSchema = serviceOutputParser.jsonSchema(returnType);
                        }
                        // 若无法使用 schema，则追加文本格式指令引导模型输出可解析结果。
                        if ((!supportsJsonSchema || jsonSchema.isEmpty()) && !streaming && !returnsImage) {
                            userMessage = appendOutputFormatInstructions(returnType, userMessage);
                        }

                        // 处理多模态占位内容，把调用参数中的 Content 合并到用户消息中。
                        Optional<List<Content>> maybeContents = findContents(method, args);
                        if (maybeContents.isPresent()) {
                            List<Content> allContents = new ArrayList<>();
                            for (Content content : maybeContents.get()) {
                                if (content == null) { // 占位符
                                    // null 作为占位符时，表示插入当前用户消息已有内容。
                                    allContents.addAll(userMessage.contents());
                                } else {
                                    // 非占位项直接追加到最终内容列表。
                                    allContents.add(content);
                                }
                            }
                            // 使用合并后的内容重建用户消息，避免原对象被就地修改。
                            userMessage = userMessage.toBuilder().contents(allContents).build();
                        }

                        // 组装本轮实际发送给模型的消息列表，兼容有/无记忆两种模式。
                        List<ChatMessage> messages = new ArrayList<>();
                        if (context.hasChatMemory()) {
                            // 有记忆时先写入 system，再拼接完整会话历史。
                            systemMessage.ifPresent(chatMemory::add);
                            messages.addAll(chatMemory.messages());
                            if (context.storeRetrievedContentInChatMemory) {
                                // 配置允许时，把增强后的用户消息写入记忆。
                                chatMemory.add(userMessage);
                            } else {
                                // 否则保留原始用户消息，避免增强内容污染长期记忆。
                                chatMemory.add(originalUserMessage);
                            }
                            // 当前轮次请求始终追加实际要发送的用户消息。
                            messages.add(userMessage);
                        } else {
                            // 无记忆时仅拼接 system + 当前用户消息。
                            systemMessage.ifPresent(messages::add);
                            messages.add(userMessage);
                        }

                        // 若方法声明需要审核，则异步触发 moderation，后续在模型返回后统一校验。
                        Future<Moderation> moderationFuture = triggerModerationIfNeeded(method, messages);

                        // 为工具调用循环预构建工具上下文（规格、执行器与错误策略等）。
                        ToolServiceContext toolServiceContext =
                                context.toolService.createContext(invocationContext, userMessage);

                        if (streaming) {
                            // 流式返回时先返回 TokenStream，模型调用在 TokenStream.start() 中触发。
                            var tokenStreamParameters = AiServiceTokenStreamParameters.builder()
                                    .messages(messages)
                                    .toolSpecifications(toolServiceContext.toolSpecifications())
                                    .toolExecutors(toolServiceContext.toolExecutors())
                                    .toolArgumentsErrorHandler(context.toolService.argumentsErrorHandler())
                                    .toolExecutionErrorHandler(context.toolService.executionErrorHandler())
                                    .toolExecutor(context.toolService.executor())
                                    .retrievedContents(
                                            augmentationResult != null ? augmentationResult.contents() : null)
                                    .context(context)
                                    .invocationContext(invocationContext)
                                    .commonGuardrailParams(commonGuardrailParam)
                                    .methodKey(method)
                                    .build();

                            // 封装为 TokenStream，允许调用方按需注册回调并延迟启动。
                            TokenStream tokenStream = new AiServiceTokenStream(tokenStreamParameters);
                            // TODO 流式场景下补齐 moderation 处理策略。
                            if (returnType == TokenStream.class) {
                                // 返回类型本身就是 TokenStream 时直接返回。
                                return tokenStream;
                            } else {
                                // 否则交由适配器转换为调用方声明的流式包装类型。
                                return adapt(tokenStream, returnType);
                            }
                        }

                        // 同步场景按需附加 responseFormat（例如 JSON Schema）。
                        ResponseFormat responseFormat = null;
                        if (supportsJsonSchema && jsonSchema.isPresent()) {
                            responseFormat = ResponseFormat.builder()
                                    .type(JSON)
                                    .jsonSchema(jsonSchema.get())
                                    .build();
                        }

                        // 结合方法注解、工具配置和输出格式，组装最终请求参数。
                        ChatRequestParameters parameters =
                                chatRequestParameters(method, args, toolServiceContext, responseFormat);

                        // 应用可选的请求转换器，以支持按 memoryId 做额外请求级改写。
                        ChatRequest chatRequest = context.chatRequestTransformer.apply(
                                ChatRequest.builder()
                                        .messages(messages)
                                        .parameters(parameters)
                                        .build(),
                                memoryId);

                        // 构建同步执行器并绑定观测上下文。
                        ChatExecutor chatExecutor = ChatExecutor.builder(context.chatModel)
                                .chatRequest(chatRequest)
                                .invocationContext(invocationContext)
                                .eventListenerRegistrar(context.eventListenerRegistrar)
                                .build();

                        // 发起模型调用并拿到首轮响应。
                        ChatResponse chatResponse = chatExecutor.execute();

                        // 首轮响应到达后立即上报事件，便于外部监听原始响应内容。
                        context.eventListenerRegistrar.fireEvent(AiServiceResponseReceivedEvent.builder()
                                .invocationContext(invocationContext)
                                .response(chatResponse)
                                .request(chatRequest)
                                .build());

                        // 在有审核任务时，这里阻塞校验审核结果，确保调用链一致性。
                        verifyModerationIfNeeded(moderationFuture);

                        // 判断是否需要以 Result<T> 包装内容与元信息。
                        boolean isReturnTypeResult = typeHasRawClass(returnType, Result.class);

                        // 执行“模型推理 + 工具循环”，产出聚合响应与中间执行信息。
                        ToolServiceResult toolServiceResult = context.toolService.executeInferenceAndToolsLoop(
                                context,
                                memoryId,
                                chatResponse,
                                parameters,
                                messages,
                                chatMemory,
                                invocationContext,
                                toolServiceContext,
                                isReturnTypeResult);

                        // 工具直接返回且目标类型为 Result<T> 时，直接构造结果并结束。
                        if (toolServiceResult.immediateToolReturn() && isReturnTypeResult) {
                            var result = Result.builder()
                                    .content(null)
                                    .tokenUsage(toolServiceResult.aggregateTokenUsage())
                                    .sources(augmentationResult == null ? null : augmentationResult.contents())
                                    .finishReason(TOOL_EXECUTION)
                                    .toolExecutions(toolServiceResult.toolExecutions())
                                    .intermediateResponses(toolServiceResult.intermediateResponses())
                                    .finalResponse(toolServiceResult.finalResponse())
                                    .build();

                            return fireEventAndReturn(invocationContext, result);
                        }

                        // 拿到工具循环后的聚合响应，作为输出护栏和解析的输入。
                        ChatResponse aggregateResponse = toolServiceResult.aggregateResponse();

                        // 执行输出护栏，可能返回替换后的响应对象。
                        var response = invokeOutputGuardrails(
                                context.guardrailService(),
                                method,
                                aggregateResponse,
                                chatExecutor,
                                commonGuardrailParam);

                        // 若护栏返回了可直接作为最终结果的类型，优先短路返回。
                        if (response != null) {
                            if (returnsImage && response instanceof ChatResponse cResponse) {
                                return fireEventAndReturn(invocationContext, parseImages(cResponse, returnType));
                            }

                            if (typeHasRawClass(returnType, response.getClass())) {
                                return fireEventAndReturn(invocationContext, response);
                            }
                        }

                        // 将最终 ChatResponse 解析为方法声明的业务返回类型。
                        var parsedResponse = serviceOutputParser.parse((ChatResponse) response, returnType);
                        // Result<T> 需要附带 token、sources、tool 执行轨迹等元信息。
                        var actualResponse = (isReturnTypeResult)
                                ? Result.builder()
                                        .content(parsedResponse)
                                        .tokenUsage(toolServiceResult.aggregateTokenUsage())
                                        .sources(augmentationResult == null ? null : augmentationResult.contents())
                                        .finishReason(toolServiceResult
                                                .finalResponse()
                                                .finishReason())
                                        .toolExecutions(toolServiceResult.toolExecutions())
                                        .intermediateResponses(toolServiceResult.intermediateResponses())
                                        .finalResponse(toolServiceResult.finalResponse())
                                        .build()
                                : parsedResponse;

                        // 统一发送完成事件并返回最终结果。
                        return fireEventAndReturn(invocationContext, actualResponse);
                    }

                    /**
                     * 统一发送“调用完成”事件并返回结果。
                     *
                     * @param invocationContext 本次调用上下文
                     * @param result            本次调用最终结果
                     * @return 原样返回传入结果，保证调用链返回语义一致
                     */
                    private Object fireEventAndReturn(InvocationContext invocationContext, Object result) {
                        // 通过统一出口发完成事件，避免多个返回分支遗漏可观测性埋点。
                        context.eventListenerRegistrar.fireEvent(AiServiceCompletedEvent.builder()
                                .invocationContext(invocationContext)
                                .result(result)
                                .build());
                        return result;
                    }

                    private static boolean isImage(Type returnType) {
                        Class<?> rawReturnType = getRawClass(returnType);
                        if (isImageType(rawReturnType)) {
                            return true;
                        }
                        if (Collection.class.isAssignableFrom(rawReturnType)) {
                            Class<?> genericParam = resolveFirstGenericParameterClass(returnType);
                            return genericParam != null && isImageType(genericParam);
                        }
                        return false;
                    }

                    private static Object parseImages(ChatResponse response, Type returnType) {
                        List<Image> images = response.aiMessage().images();
                        Class<?> rawReturnType = getRawClass(returnType);
                        if (isImage(rawReturnType)) {
                            if (rawReturnType == ImageContent.class) {
                                List<ImageContent> imageContents = toImageContents(images);
                                return imageContents.isEmpty() ? null : imageContents.get(0);
                            }
                            if (rawReturnType == Image.class) {
                                return images.isEmpty() ? null : images.get(0);
                            }
                        }
                        if (Collection.class.isAssignableFrom(rawReturnType)) {
                            Class<?> genericParam = resolveFirstGenericParameterClass(returnType);
                            if (genericParam == ImageContent.class) {
                                return toImageContents(images);
                            }
                            if (genericParam == Image.class) {
                                return images;
                            }
                        }
                        throw new UnsupportedOperationException("Unsupported return type " + rawReturnType);
                    }

                    private static List<ImageContent> toImageContents(List<Image> images) {
                        return images.stream().map(ImageContent::from).toList();
                    }

                    private boolean canAdaptTokenStreamTo(Type returnType) {
                        for (TokenStreamAdapter tokenStreamAdapter : tokenStreamAdapters) {
                            if (tokenStreamAdapter.canAdaptTokenStreamTo(returnType)) {
                                return true;
                            }
                        }
                        return false;
                    }

                    private Object adapt(TokenStream tokenStream, Type returnType) {
                        for (TokenStreamAdapter tokenStreamAdapter : tokenStreamAdapters) {
                            if (tokenStreamAdapter.canAdaptTokenStreamTo(returnType)) {
                                return tokenStreamAdapter.adapt(tokenStream);
                            }
                        }
                        throw new IllegalStateException("Can't find suitable TokenStreamAdapter");
                    }

                    /**
                     * 判断当前模型是否支持 JSON Schema 结构化输出能力。
                     *
                     * @return 支持返回 {@code true}，否则返回 {@code false}
                     */
                    private boolean supportsJsonSchema() {
                        // 先判空再读取能力集合，避免在仅配置流式模型等场景出现空指针。
                        return context.chatModel != null
                                && context.chatModel.supportedCapabilities().contains(RESPONSE_FORMAT_JSON_SCHEMA);
                    }

                    /**
                     * 在无法使用 JSON Schema 时，将输出格式约束追加到用户消息中。
                     *
                     * @param returnType  目标返回类型，用于推导格式约束文本
                     * @param userMessage 当前用户消息
                     * @return 追加格式约束后的用户消息；若无约束可追加则返回原消息
                     */
                    private UserMessage appendOutputFormatInstructions(Type returnType, UserMessage userMessage) {
                        // 先由解析器按返回类型生成格式说明，确保提示词与解析器预期一致。
                        String outputFormatInstructions = serviceOutputParser.outputFormatInstructions(returnType);
                        if (isNullOrEmpty(outputFormatInstructions)) {
                            // 无格式要求时直接返回，避免无意义地重建消息对象。
                            return userMessage;
                        }

                        // 复制内容列表后再改写，避免就地修改原消息带来副作用。
                        List<Content> contents = new ArrayList<>(userMessage.contents());

                        boolean appended = false;
                        // 倒序查找最后一个文本内容，优先把约束拼接到已有文本末尾，减少内容片段割裂。
                        for (int i = contents.size() - 1; i >= 0; i--) {
                            if (contents.get(i) instanceof TextContent lastTextContent) {
                                String newText = lastTextContent.text() + outputFormatInstructions;
                                contents.set(i, TextContent.from(newText));
                                appended = true;
                                break;
                            }
                        }

                        if (!appended) {
                            // 若不存在文本片段，则新建文本内容承载输出格式要求。
                            contents.add(TextContent.from(outputFormatInstructions));
                        }

                        // 返回新消息对象，保证不可变语义并便于后续链路继续加工。
                        return userMessage.toBuilder().contents(contents).build();
                    }

                    /**
                     * 按方法注解决定是否异步触发审核任务。
                     *
                     * @param method   当前调用的方法
                     * @param messages 本轮待审核消息列表
                     * @return 审核任务 Future；若无需审核则返回 {@code null}
                     */
                    private Future<Moderation> triggerModerationIfNeeded(Method method, List<ChatMessage> messages) {
                        if (method.isAnnotationPresent(Moderate.class)) {
                            // 审核放入线程池异步执行，尽量与模型请求并行以降低总时延。
                            ExecutorService executor = DefaultExecutorProvider.getDefaultExecutorService();
                            return executor.submit(() -> {
                                // 工具消息通常包含执行回显，不应作为内容审核输入。
                                List<ChatMessage> messagesToModerate = removeToolMessages(messages);
                                return context.moderationModel
                                        .moderate(messagesToModerate)
                                        .content();
                            });
                        }
                        return null;
                    }
                });

        // 动态代理已完成构建，这里做受控类型转换返回调用方。
        return (T) proxyInstance;
    }

    /**
     * 执行输入护栏，必要时替换或拒绝用户输入消息。
     *
     * @param guardrailService      护栏服务
     * @param method                当前调用的方法
     * @param userMessage           当前用户消息
     * @param commonGuardrailParams 输入/输出护栏共享参数
     * @return 护栏处理后的用户消息
     */
    private UserMessage invokeInputGuardrails(
            GuardrailService guardrailService,
            Method method,
            UserMessage userMessage,
            GuardrailRequestParams commonGuardrailParams) {

        // 是否存在输入护栏通常带缓存，调用前检查可避免每次都构建请求对象。
        if (guardrailService.hasInputGuardrails(method)) {
            // 统一封装输入护栏请求，确保护栏插件拿到完整上下文。
            var inputGuardrailRequest = InputGuardrailRequest.builder()
                    .userMessage(userMessage)
                    .commonParams(commonGuardrailParams)
                    .build();
            return guardrailService.executeGuardrails(method, inputGuardrailRequest);
        }

        // 无输入护栏时保持原消息，避免引入不必要的处理开销。
        return userMessage;
    }

    /**
     * 执行输出护栏，必要时对模型响应进行拦截或替换。
     *
     * @param guardrailService      护栏服务
     * @param method                当前调用的方法
     * @param responseFromLLM       模型原始响应
     * @param chatExecutor          可用于护栏内触发重试/再请求的执行器
     * @param commonGuardrailParams 输入/输出护栏共享参数
     * @param <T>                   护栏返回类型
     * @return 护栏处理结果；无输出护栏时返回原始响应
     */
    private <T> T invokeOutputGuardrails(
            GuardrailService guardrailService,
            Method method,
            ChatResponse responseFromLLM,
            ChatExecutor chatExecutor,
            GuardrailRequestParams commonGuardrailParams) {

        if (guardrailService.hasOutputGuardrails(method)) {
            // 输出护栏需要原始响应与执行器，才能在策略允许时执行“校验失败后重试”。
            var outputGuardrailRequest = OutputGuardrailRequest.builder()
                    .responseFromLLM(responseFromLLM)
                    .chatExecutor(chatExecutor)
                    .requestParams(commonGuardrailParams)
                    .build();
            return guardrailService.executeGuardrails(method, outputGuardrailRequest);
        }

        // 未配置输出护栏时直接透传模型响应。
        return (T) responseFromLLM;
    }

    /**
     * 根据系统消息模板渲染本轮系统消息。
     *
     * @param memoryId 会话内存标识
     * @param method   当前调用的方法
     * @param args     方法参数
     * @return 渲染后的系统消息；未找到模板时返回空
     */
    private Optional<SystemMessage> prepareSystemMessage(Object memoryId, Method method, Object[] args) {
        // 先定位模板来源，再统一走模板引擎渲染，保证变量替换行为一致。
        return findSystemMessageTemplate(memoryId, method).map(systemMessageTemplate -> PromptTemplate.from(
                        systemMessageTemplate)
                .apply(InternalReflectionVariableResolver.findTemplateVariables(systemMessageTemplate, method, args))
                .toSystemMessage());
    }

    /**
     * 查找系统消息模板，优先方法注解，其次上下文提供器。
     *
     * @param memoryId 会话内存标识
     * @param method   当前调用的方法
     * @return 系统消息模板文本
     */
    private Optional<String> findSystemMessageTemplate(Object memoryId, Method method) {
        dev.langchain4j.service.SystemMessage annotation =
                method.getAnnotation(dev.langchain4j.service.SystemMessage.class);
        if (annotation != null) {
            // 方法注解优先级最高，保证调用方可在方法级覆盖全局模板提供器。
            return Optional.of(getTemplate(
                    method, "System", annotation.fromResource(), annotation.value(), annotation.delimiter()));
        }

        // 未声明注解时回退到 memoryId 维度的模板提供器，支持多租户/多会话差异化 system prompt。
        return context.systemMessageProvider.apply(memoryId);
    }

    /**
     * 组装用户消息：可来自模板渲染，也可直接由多模态内容参数构建。
     *
     * @param method              当前调用的方法
     * @param args                方法参数
     * @param userMessageTemplate 用户消息模板
     * @param variables           模板变量
     * @return 可发送给模型的用户消息
     * @throws IllegalConfigurationException 当方法既无模板也无可用内容参数时抛出
     */
    private static UserMessage prepareUserMessage(
            Method method, Object[] args, String userMessageTemplate, Map<String, Object> variables) {

        // 用户名是可选元信息，若存在则用于构造带“发言人”语义的 UserMessage。
        Optional<String> maybeUserName = findUserName(method.getParameters(), args);

        if (userMessageTemplate.isEmpty()) {
            // 模板为空表示本轮走“纯内容模式”，从参数中抽取 Content / List<Content>。
            List<Content> contents = new ArrayList<>();

            for (Object arg : args) {
                if (arg instanceof Content content) {
                    contents.add(content);
                } else if (isListOfContents(arg)) {
                    contents.addAll((List<Content>) arg);
                }
            }

            if (!contents.isEmpty()) {
                // 有内容时优先保留 userName，便于下游按用户名进行策略处理。
                return maybeUserName
                        .map(userName -> UserMessage.from(userName, contents))
                        .orElseGet(() -> UserMessage.from(contents));
            }

            // 既无模板又无内容参数时无法生成有效用户消息，应尽早以配置错误失败。
            throw illegalConfiguration(
                    "Error: The method '%s' does not have a user message defined.", method.getName());
        }

        // 模板模式下先渲染文本，再根据是否携带 userName 构造最终 UserMessage。
        Prompt prompt = PromptTemplate.from(userMessageTemplate).apply(variables);

        return maybeUserName
                .map(userName -> UserMessage.from(userName, prompt.text()))
                .orElseGet(prompt::toUserMessage);
    }

    private String getUserMessageTemplate(Object memoryId, Method method, Object[] args) {

        Optional<String> templateFromMethodAnnotation = findUserMessageTemplateFromMethodAnnotation(method);
        Optional<String> templateFromParameterAnnotation =
                findUserMessageTemplateFromAnnotatedParameter(method.getParameters(), args);

        if (templateFromMethodAnnotation.isPresent() && templateFromParameterAnnotation.isPresent()) {
            throw illegalConfiguration(
                    "Error: The method '%s' has multiple @UserMessage annotations. Please use only one.",
                    method.getName());
        }

        if (templateFromMethodAnnotation.isPresent()) {
            return templateFromMethodAnnotation.get();
        }
        if (templateFromParameterAnnotation.isPresent()) {
            return templateFromParameterAnnotation.get();
        }

        Optional<String> templateFromTheOnlyArgument =
                findUserMessageTemplateFromTheOnlyArgument(method.getParameters(), args);
        if (templateFromTheOnlyArgument.isPresent()) {
            return templateFromTheOnlyArgument.get();
        }

        if (hasContentArgument(method, args)) {
            return "";
        }

        return context.userMessageProvider.apply(memoryId)
                .orElseThrow(() -> illegalConfiguration("Error: The method '%s' does not have a user message defined.", method.getName()));
    }

    private static boolean hasContentArgument(Method method, Object[] args) {
        Parameter[] parameters = method.getParameters();
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(dev.langchain4j.service.UserMessage.class)) {
                if (args[i] instanceof Content || isListOfContents(args[i])) {
                    return true;
                }
            }
        }

        if (parameters.length == 1 && !hasAnyValidAnnotation(parameters[0])) {
            return args[0] instanceof Content || isListOfContents(args[0]);
        }
        return false;
    }

    private static Optional<String> findUserMessageTemplateFromMethodAnnotation(Method method) {
        return Optional.ofNullable(method.getAnnotation(dev.langchain4j.service.UserMessage.class))
                .map(a -> getTemplate(method, "User", a.fromResource(), a.value(), a.delimiter()));
    }

    private static Optional<String> findUserMessageTemplateFromAnnotatedParameter(
            Parameter[] parameters, Object[] args) {
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(dev.langchain4j.service.UserMessage.class)
                    && !(args[i] instanceof Content)
                    && !isListOfContents(args[i])) {
                return Optional.of(InternalReflectionVariableResolver.asString(args[i]));
            }
        }
        return Optional.empty();
    }

    private static boolean hasAnyValidAnnotation(Parameter parameter) {
        for (Class<? extends Annotation> a : VALID_PARAM_ANNOTATIONS) {
            if (parameter.getAnnotation(a) != null) {
                return true;
            }
        }

        return false;
    }

    private static Optional<String> findUserMessageTemplateFromTheOnlyArgument(Parameter[] parameters, Object[] args) {
        if (parameters != null && parameters.length == 1 && !hasAnyValidAnnotation(parameters[0])) {
            if (args[0] instanceof Content || isListOfContents(args[0])) {
                return Optional.empty();
            }
            return Optional.of(InternalReflectionVariableResolver.asString(args[0]));
        }
        return Optional.empty();
    }

    /**
     * 从方法参数中提取 {@link UserName} 标注的用户名。
     *
     * @param parameters 方法参数定义
     * @param args       方法参数值
     * @return 用户名；未声明 {@link UserName} 时返回空
     */
    private static Optional<String> findUserName(Parameter[] parameters, Object[] args) {
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(UserName.class)) {
                // 按注解定位用户名参数，统一转字符串以兼容多种入参类型。
                return Optional.of(args[i].toString());
            }
        }
        return Optional.empty();
    }

    /**
     * 从方法参数中收集多模态内容列表，并处理文本占位符规则。
     *
     * @param method 当前调用的方法
     * @param args   方法参数值
     * @return 内容列表；不存在内容参数时返回空
     * @throws IllegalConfigurationException 当文本占位符来源超过一个时抛出
     */
    private static Optional<List<Content>> findContents(Method method, Object[] args) {
        List<Content> contents = new ArrayList<>();

        if (findUserMessageTemplateFromMethodAnnotation(method).isPresent()) {
            // 方法级模板存在时预置一个文本占位符，后续由调用链替换为实际文本消息内容。
            contents.add(null); // placeholder
        }

        Parameter[] parameters = method.getParameters();
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(dev.langchain4j.service.UserMessage.class)) {
                if (args[i] instanceof Content) {
                    contents.add((Content) args[i]);
                } else if (isListOfContents(args[i])) {
                    contents.addAll((List<Content>) args[i]);
                } else {
                    // 被 @UserMessage 标注但不是 Content 时，视为文本占位来源。
                    contents.add(null); // placeholder
                }
            }
        }

        if (contents.isEmpty() && parameters.length == 1 && !hasAnyValidAnnotation(parameters[0])) {
            // 单参数且无特殊注解时允许隐式把参数当作 Content 入参，兼容简写方法签名。
            if (args[0] instanceof Content) {
                contents.add((Content) args[0]);
            } else if (isListOfContents(args[0])) {
                contents.addAll((List<Content>) args[0]);
            }
        }

        if (contents.stream().filter(Objects::isNull).count() > 1) {
            // 文本占位符只能有一个，否则无法确定最终文本插入位置与来源。
            throw illegalConfiguration(
                    "Error: The method '%s' has multiple @UserMessage for text content. Please use only one.",
                    method.getName());
        }

        // 仅在收集到内容时返回，避免上游误判为空列表也代表“有内容”。
        return contents.isEmpty() ? Optional.empty() : Optional.of(contents);
    }

    private static boolean isListOfContents(Object o) {
        return o instanceof List<?> list && list.stream().allMatch(Content.class::isInstance);
    }

    private static String getTemplate(Method method, String type, String resource, String[] value, String delimiter) {
        String messageTemplate;
        if (!resource.trim().isEmpty()) {
            messageTemplate = getResourceText(method.getDeclaringClass(), resource);
            if (messageTemplate == null) {
                throw illegalConfiguration("@%sMessage's resource '%s' not found", type, resource);
            }
        } else {
            messageTemplate = String.join(delimiter, value);
        }
        if (messageTemplate.trim().isEmpty()) {
            throw illegalConfiguration("@%sMessage's template cannot be empty", type);
        }
        return messageTemplate;
    }

    private static String getResourceText(Class<?> clazz, String resource) {
        InputStream inputStream = clazz.getResourceAsStream(resource);
        if (inputStream == null) {
            inputStream = clazz.getResourceAsStream("/" + resource);
        }
        return getText(inputStream);
    }

    private static String getText(InputStream inputStream) {
        if (inputStream == null) {
            return null;
        }
        try (Scanner scanner = new Scanner(inputStream);
                Scanner s = scanner.useDelimiter("\\A")) {
            return s.hasNext() ? s.next() : "";
        }
    }

    /**
     * 提取方法参数中的 {@link MemoryId} 值，作为会话记忆分片键。
     *
     * @param method 当前调用的方法
     * @param args   方法参数值
     * @return memoryId；未声明 {@link MemoryId} 时返回空
     * @throws IllegalArgumentException 当 {@link MemoryId} 参数值为 {@code null} 时抛出
     */
    private static Optional<Object> findMemoryId(Method method, Object[] args) {
        Parameter[] parameters = method.getParameters();
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(MemoryId.class)) {
                Object memoryId = args[i];
                if (memoryId == null) {
                    // memoryId 参与会话隔离，空值会导致记忆路由不确定，必须显式拒绝。
                    throw illegalArgument(
                            "The value of parameter '%s' annotated with @MemoryId in method '%s' must not be null",
                            parameters[i].getName(), method.getName());
                }
                return Optional.of(memoryId);
            }
        }
        return Optional.empty();
    }
}
