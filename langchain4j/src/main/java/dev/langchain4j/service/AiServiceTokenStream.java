package dev.langchain4j.service;

import static dev.langchain4j.internal.Utils.copy;
import static dev.langchain4j.internal.ValidationUtils.ensureNotEmpty;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;
import static dev.langchain4j.service.AiServiceParamsUtil.chatRequestParameters;

import dev.langchain4j.Internal;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.guardrail.ChatExecutor;
import dev.langchain4j.guardrail.GuardrailRequestParams;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.PartialToolCallContext;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.observability.api.event.AiServiceRequestIssuedEvent;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.service.tool.BeforeToolExecution;
import dev.langchain4j.service.tool.ToolArgumentsErrorHandler;
import dev.langchain4j.service.tool.ToolExecution;
import dev.langchain4j.service.tool.ToolExecutionErrorHandler;
import dev.langchain4j.service.tool.ToolExecutor;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

@Internal
public class AiServiceTokenStream implements TokenStream {

    private final List<ChatMessage> messages;

    private final List<ToolSpecification> toolSpecifications;
    private final Map<String, ToolExecutor> toolExecutors;
    private final ToolArgumentsErrorHandler toolArgumentsErrorHandler;
    private final ToolExecutionErrorHandler toolExecutionErrorHandler;
    private final Executor toolExecutor;

    private final List<Content> retrievedContents;
    private final AiServiceContext context;
    private final InvocationContext invocationContext;
    private final GuardrailRequestParams commonGuardrailParams;
    private final Object methodKey;

    private Consumer<String> partialResponseHandler;
    private BiConsumer<PartialResponse, PartialResponseContext> partialResponseWithContextHandler;
    private Consumer<PartialThinking> partialThinkingHandler;
    private BiConsumer<PartialThinking, PartialThinkingContext> partialThinkingWithContextHandler;
    private Consumer<PartialToolCall> partialToolCallHandler;
    private BiConsumer<PartialToolCall, PartialToolCallContext> partialToolCallWithContextHandler;
    private Consumer<List<Content>> contentsHandler;
    private Consumer<ChatResponse> intermediateResponseHandler;
    private Consumer<BeforeToolExecution> beforeToolExecutionHandler;
    private Consumer<ToolExecution> toolExecutionHandler;
    private Consumer<ChatResponse> completeResponseHandler;
    private Consumer<Throwable> errorHandler;

    private int onPartialResponseInvoked;
    private int onPartialResponseWithContextInvoked;
    private int onPartialThinkingInvoked;
    private int onPartialThinkingWithContextInvoked;
    private int onPartialToolCallInvoked;
    private int onPartialToolCallWithContextInvoked;
    private int onIntermediateResponseInvoked;
    private int onCompleteResponseInvoked;
    private int onRetrievedInvoked;
    private int beforeToolExecutionInvoked;
    private int onToolExecutedInvoked;
    private int onErrorInvoked;
    private int ignoreErrorsInvoked;

    /**
     * 使用给定参数创建 {@link AiServiceTokenStream} 实例。
     *
     * @param parameters 构建流式处理器所需的参数集合
     */
    public AiServiceTokenStream(AiServiceTokenStreamParameters parameters) {
        ensureNotNull(parameters, "parameters");
        this.messages = copy(ensureNotEmpty(parameters.messages(), "messages"));
        this.toolSpecifications = copy(parameters.toolSpecifications());
        this.toolExecutors = copy(parameters.toolExecutors());
        this.toolArgumentsErrorHandler = parameters.toolArgumentsErrorHandler();
        this.toolExecutionErrorHandler = parameters.toolExecutionErrorHandler();
        this.toolExecutor = parameters.toolExecutor();
        this.retrievedContents = copy(parameters.retrievedContents());
        this.context = ensureNotNull(parameters.context(), "context");
        ensureNotNull(this.context.streamingChatModel, "streamingChatModel");
        this.invocationContext = parameters.invocationContext();
        this.commonGuardrailParams = parameters.commonGuardrailParams();
        this.methodKey = parameters.methodKey();
    }

    @Override
    public TokenStream onPartialResponse(Consumer<String> partialResponseHandler) {
        this.partialResponseHandler = partialResponseHandler;
        this.onPartialResponseInvoked++;
        return this;
    }

    @Override
    public TokenStream onPartialResponseWithContext(BiConsumer<PartialResponse, PartialResponseContext> handler) {
        this.partialResponseWithContextHandler = handler;
        this.onPartialResponseWithContextInvoked++;
        return this;
    }

    @Override
    public TokenStream onPartialThinking(Consumer<PartialThinking> partialThinkingHandler) {
        this.partialThinkingHandler = partialThinkingHandler;
        this.onPartialThinkingInvoked++;
        return this;
    }

    @Override
    public TokenStream onPartialThinkingWithContext(BiConsumer<PartialThinking, PartialThinkingContext> handler) {
        this.partialThinkingWithContextHandler = handler;
        this.onPartialThinkingWithContextInvoked++;
        return this;
    }

    @Override
    public TokenStream onPartialToolCall(Consumer<PartialToolCall> partialToolCallHandler) {
        this.partialToolCallHandler = partialToolCallHandler;
        this.onPartialToolCallInvoked++;
        return this;
    }

    @Override
    public TokenStream onPartialToolCallWithContext(BiConsumer<PartialToolCall, PartialToolCallContext> handler) {
        this.partialToolCallWithContextHandler = handler;
        this.onPartialToolCallWithContextInvoked++;
        return this;
    }

    @Override
    public TokenStream onRetrieved(Consumer<List<Content>> contentsHandler) {
        this.contentsHandler = contentsHandler;
        this.onRetrievedInvoked++;
        return this;
    }

    @Override
    public TokenStream onIntermediateResponse(Consumer<ChatResponse> intermediateResponseHandler) {
        this.intermediateResponseHandler = intermediateResponseHandler;
        this.onIntermediateResponseInvoked++;
        return this;
    }

    @Override
    public TokenStream beforeToolExecution(Consumer<BeforeToolExecution> beforeToolExecutionHandler) {
        this.beforeToolExecutionHandler = beforeToolExecutionHandler;
        this.beforeToolExecutionInvoked++;
        return this;
    }

    @Override
    public TokenStream onToolExecuted(Consumer<ToolExecution> toolExecutionHandler) {
        this.toolExecutionHandler = toolExecutionHandler;
        this.onToolExecutedInvoked++;
        return this;
    }

    @Override
    public TokenStream onCompleteResponse(Consumer<ChatResponse> completionHandler) {
        this.completeResponseHandler = completionHandler;
        this.onCompleteResponseInvoked++;
        return this;
    }

    @Override
    public TokenStream onError(Consumer<Throwable> errorHandler) {
        this.errorHandler = errorHandler;
        this.onErrorInvoked++;
        return this;
    }

    @Override
    public TokenStream ignoreErrors() {
        this.errorHandler = null;
        this.ignoreErrorsInvoked++;
        return this;
    }

    /**
     * 启动流式请求执行流程。
     * 该方法会构造 {@link ChatRequest} 与 {@link AiServiceStreamingResponseHandler}，
     * 然后将处理器传给底层 {@link dev.langchain4j.model.chat.StreamingChatModel}。
     */
    @Override
    public void start() {
        // 启动前先校验回调配置，避免运行中出现“缺少 onError/重复注册”等问题。
        validateConfiguration();

        // 将调用上下文转换为最终 ChatRequest，统一应用请求参数与 memoryId 变换。
        ChatRequest chatRequest = context.chatRequestTransformer.apply(
                ChatRequest.builder()
                        .messages(messages)
                        .parameters(chatRequestParameters(invocationContext.methodArguments(), toolSpecifications))
                        .build(),
                invocationContext.chatMemoryId());

        // 构造流式执行器，便于在工具循环和护栏流程中复用执行上下文。
        ChatExecutor chatExecutor = ChatExecutor.builder(context.streamingChatModel)
                .errorHandler(errorHandler)
                .chatRequest(chatRequest)
                .invocationContext(invocationContext)
                .eventListenerRegistrar(context.eventListenerRegistrar)
                .build();

        // 组装 AI Service 层的统一流式 handler，承接 partial/complete/error 与工具调用续轮逻辑。
        var handler = new AiServiceStreamingResponseHandler(
                chatRequest,
                chatExecutor,
                context,
                invocationContext,
                partialResponseHandler,
                partialResponseWithContextHandler,
                partialThinkingHandler,
                partialThinkingWithContextHandler,
                partialToolCallHandler,
                partialToolCallWithContextHandler,
                beforeToolExecutionHandler,
                toolExecutionHandler,
                intermediateResponseHandler,
                completeResponseHandler,
                errorHandler,
                initTemporaryMemory(context, messages),
                new TokenUsage(),
                toolSpecifications,
                toolExecutors,
                context.toolService.maxSequentialToolsInvocations(),
                toolArgumentsErrorHandler,
                toolExecutionErrorHandler,
                toolExecutor,
                commonGuardrailParams,
                methodKey);

        // 若配置了检索内容回调，则在请求发出前把检索结果先回调给调用方。
        if (contentsHandler != null && retrievedContents != null) {
            contentsHandler.accept(retrievedContents);
        }

        // 在真正触发模型调用前发送“请求已发出”事件，方便观测系统记录请求轨迹。
        context.eventListenerRegistrar.fireEvent(AiServiceRequestIssuedEvent.builder()
                .invocationContext(invocationContext)
                .request(chatRequest)
                .build());

        // 这里是流式调用的关键跳转点：将上层回调封装成 handler 后交给具体模型实现。
        context.streamingChatModel.chat(chatRequest, handler);
    }

    /**
     * 校验流式回调配置的合法性。
     * 该校验在真正发起模型调用前执行，用于尽早发现“重复注册”和“缺失必需配置”两类错误，
     * 避免链路运行到中途才抛错而导致上下文状态不一致。
     */
    private void validateConfiguration() {
        // partialResponse 仅允许二选一，防止同一 token 被重复分发。
        if (onPartialResponseInvoked + onPartialResponseWithContextInvoked > 1) {
            throw new IllegalConfigurationException("One of [onPartialResponse, onPartialResponseWithContext] "
                    + "can be invoked on TokenStream at most 1 time");
        }
        // partialThinking 同样要求二选一，确保思维流回调语义稳定且唯一。
        if (onPartialThinkingInvoked + onPartialThinkingWithContextInvoked > 1) {
            throw new IllegalConfigurationException("One of [onPartialThinking, onPartialThinkingWithContext] "
                    + "can be invoked on TokenStream at most 1 time");
        }
        // partialToolCall 二选一，避免工具调用片段在不同回调通道重复消费。
        if (onPartialToolCallInvoked + onPartialToolCallWithContextInvoked > 1) {
            throw new IllegalConfigurationException("One of [onPartialToolCall, onPartialToolCallWithContext] can be "
                    + "invoked on TokenStream at most 1 time");
        }
        // 中间响应只能绑定一次，避免工具轮次之间出现多重分发副作用。
        if (onIntermediateResponseInvoked > 1) {
            throw new IllegalConfigurationException(
                    "onIntermediateResponse can be invoked on TokenStream at most 1 time");
        }
        // 完成回调只允许一次，保证最终响应只会进入单一消费端。
        if (onCompleteResponseInvoked > 1) {
            throw new IllegalConfigurationException("onCompleteResponse can be invoked on TokenStream at most 1 time");
        }
        // 检索内容回调应唯一，避免同一批 RAG 内容被重复通知。
        if (onRetrievedInvoked > 1) {
            throw new IllegalConfigurationException("onRetrieved can be invoked on TokenStream at most 1 time");
        }
        // 工具执行前回调仅允许一个消费者，避免预处理逻辑重复触发。
        if (beforeToolExecutionInvoked > 1) {
            throw new IllegalConfigurationException("beforeToolExecution can be invoked on TokenStream at most 1 time");
        }
        // 工具执行后回调同样只允许一次，保证观测与埋点口径一致。
        if (onToolExecutedInvoked > 1) {
            throw new IllegalConfigurationException("onToolExecuted can be invoked on TokenStream at most 1 time");
        }
        // 错误策略必须显式二选一：要么上抛给 onError，要么声明忽略，禁止隐式默认行为。
        if (onErrorInvoked + ignoreErrorsInvoked != 1) {
            throw new IllegalConfigurationException(
                    "One of [onError, ignoreErrors] " + "must be invoked on TokenStream exactly 1 time");
        }
    }

    /**
     * 初始化工具循环阶段使用的临时记忆。
     * 当外部未配置持久 {@link ChatMemory} 时，临时记忆需要承接当前请求消息，
     * 以便工具执行后续轮请求仍能读取完整上下文。
     *
     * @param context AI Service 运行上下文
     * @param messagesToSend 当前首轮将发送给模型的消息
     * @return 可用于本次流式链路的记忆实例
     */
    private ChatMemory initTemporaryMemory(AiServiceContext context, List<ChatMessage> messagesToSend) {
        // 这里使用“无限窗口”作为临时容器，避免工具多轮调用时上下文被意外截断。
        var chatMemory = MessageWindowChatMemory.withMaxMessages(Integer.MAX_VALUE);

        if (!context.hasChatMemory()) {
            // 仅在没有外部记忆时回灌首轮消息，防止覆盖外部记忆服务中的既有状态。
            chatMemory.add(messagesToSend);
        }

        // 无论是否回灌消息，都返回统一的临时记忆对象供后续 handler 使用。
        return chatMemory;
    }
}
