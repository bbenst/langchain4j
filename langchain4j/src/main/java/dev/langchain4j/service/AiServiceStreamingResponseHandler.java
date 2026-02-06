package dev.langchain4j.service;

import static dev.langchain4j.internal.Exceptions.runtime;
import static dev.langchain4j.internal.Utils.copy;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;
import static dev.langchain4j.service.AiServiceParamsUtil.chatRequestParameters;

import dev.langchain4j.Internal;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.guardrail.ChatExecutor;
import dev.langchain4j.guardrail.GuardrailRequestParams;
import dev.langchain4j.guardrail.OutputGuardrailRequest;
import dev.langchain4j.invocation.InvocationContext;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.PartialToolCall;
import dev.langchain4j.model.chat.response.PartialToolCallContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.observability.api.event.AiServiceCompletedEvent;
import dev.langchain4j.observability.api.event.AiServiceErrorEvent;
import dev.langchain4j.observability.api.event.AiServiceRequestIssuedEvent;
import dev.langchain4j.observability.api.event.AiServiceResponseReceivedEvent;
import dev.langchain4j.observability.api.event.ToolExecutedEvent;
import dev.langchain4j.service.tool.BeforeToolExecution;
import dev.langchain4j.service.tool.ToolArgumentsErrorHandler;
import dev.langchain4j.service.tool.ToolExecution;
import dev.langchain4j.service.tool.ToolExecutionErrorHandler;
import dev.langchain4j.service.tool.ToolExecutionResult;
import dev.langchain4j.service.tool.ToolExecutor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Future;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * AI Service 的流式响应处理器。
 * 负责消费模型逐 token 返回的数据，并在“普通文本响应”和“工具调用响应”两种路径之间切换，
 * 同时维护事件上报、记忆写入、工具执行与续轮请求的完整闭环。
 */
@Internal
class AiServiceStreamingResponseHandler implements StreamingChatResponseHandler {

    private static final Logger LOG = LoggerFactory.getLogger(AiServiceStreamingResponseHandler.class);

    private final ChatExecutor chatExecutor;
    private final ChatRequest chatRequest;
    private final AiServiceContext context;
    private final InvocationContext invocationContext;
    private final GuardrailRequestParams commonGuardrailParams;
    private final Object methodKey;

    private final Consumer<String> partialResponseHandler;
    private final BiConsumer<PartialResponse, PartialResponseContext> partialResponseWithContextHandler;
    private final Consumer<PartialThinking> partialThinkingHandler;
    private final BiConsumer<PartialThinking, PartialThinkingContext> partialThinkingWithContextHandler;
    private final Consumer<PartialToolCall> partialToolCallHandler;
    private final BiConsumer<PartialToolCall, PartialToolCallContext> partialToolCallWithContextHandler;
    private final Consumer<BeforeToolExecution> beforeToolExecutionHandler;
    private final Consumer<ToolExecution> toolExecutionHandler;
    private final Consumer<ChatResponse> intermediateResponseHandler;
    private final Consumer<ChatResponse> completeResponseHandler;

    private final Consumer<Throwable> errorHandler;

    private final ChatMemory temporaryMemory;
    private final TokenUsage tokenUsage;

    private final List<ToolSpecification> toolSpecifications;
    private final Map<String, ToolExecutor> toolExecutors;
    private final ToolArgumentsErrorHandler toolArgumentsErrorHandler;
    private final ToolExecutionErrorHandler toolExecutionErrorHandler;
    private final Executor toolExecutor;
    private final Queue<Future<ToolRequestResult>> toolExecutionFutures = new ConcurrentLinkedQueue<>();

    private final List<String> responseBuffer = new ArrayList<>();
    private final boolean hasOutputGuardrails;

    private int sequentialToolsInvocationsLeft;

    private record ToolRequestResult(ToolExecutionRequest request, ToolExecutionResult result) {}

    AiServiceStreamingResponseHandler(
            ChatRequest chatRequest,
            ChatExecutor chatExecutor,
            AiServiceContext context,
            InvocationContext invocationContext,
            Consumer<String> partialResponseHandler,
            BiConsumer<PartialResponse, PartialResponseContext> partialResponseWithContextHandler,
            Consumer<PartialThinking> partialThinkingHandler,
            BiConsumer<PartialThinking, PartialThinkingContext> partialThinkingWithContextHandler,
            Consumer<PartialToolCall> partialToolCallHandler,
            BiConsumer<PartialToolCall, PartialToolCallContext> partialToolCallWithContextHandler,
            Consumer<BeforeToolExecution> beforeToolExecutionHandler,
            Consumer<ToolExecution> toolExecutionHandler,
            Consumer<ChatResponse> intermediateResponseHandler,
            Consumer<ChatResponse> completeResponseHandler,
            Consumer<Throwable> errorHandler,
            ChatMemory temporaryMemory,
            TokenUsage tokenUsage,
            List<ToolSpecification> toolSpecifications,
            Map<String, ToolExecutor> toolExecutors,
            int sequentialToolsInvocationsLeft,
            ToolArgumentsErrorHandler toolArgumentsErrorHandler,
            ToolExecutionErrorHandler toolExecutionErrorHandler,
            Executor toolExecutor,
            GuardrailRequestParams commonGuardrailParams,
            Object methodKey) {
        this.chatRequest = ensureNotNull(chatRequest, "chatRequest");
        this.chatExecutor = ensureNotNull(chatExecutor, "chatExecutor");
        this.context = ensureNotNull(context, "context");
        this.invocationContext = ensureNotNull(invocationContext, "invocationContext");
        this.methodKey = methodKey;

        this.partialResponseHandler = partialResponseHandler;
        this.partialResponseWithContextHandler = partialResponseWithContextHandler;
        this.partialThinkingHandler = partialThinkingHandler;
        this.partialThinkingWithContextHandler = partialThinkingWithContextHandler;
        this.partialToolCallHandler = partialToolCallHandler;
        this.partialToolCallWithContextHandler = partialToolCallWithContextHandler;
        this.intermediateResponseHandler = intermediateResponseHandler;
        this.completeResponseHandler = completeResponseHandler;
        this.beforeToolExecutionHandler = beforeToolExecutionHandler;
        this.toolExecutionHandler = toolExecutionHandler;
        this.errorHandler = errorHandler;

        this.temporaryMemory = temporaryMemory;
        this.tokenUsage = ensureNotNull(tokenUsage, "tokenUsage");
        this.commonGuardrailParams = commonGuardrailParams;

        this.toolSpecifications = copy(toolSpecifications);
        this.toolExecutors = copy(toolExecutors);
        this.toolArgumentsErrorHandler = ensureNotNull(toolArgumentsErrorHandler, "toolArgumentsErrorHandler");
        this.toolExecutionErrorHandler = ensureNotNull(toolExecutionErrorHandler, "toolExecutionErrorHandler");
        this.toolExecutor = toolExecutor;

        this.hasOutputGuardrails = context.guardrailService().hasOutputGuardrails(methodKey);

        this.sequentialToolsInvocationsLeft = sequentialToolsInvocationsLeft;
    }

    @Override
    public void onPartialResponse(String partialResponse) {
        // 启用输出护栏时先缓存 partial，待护栏完成后再统一对外回放。
        if (hasOutputGuardrails) {
            responseBuffer.add(partialResponse);
        } else if (partialResponseHandler != null) {
            partialResponseHandler.accept(partialResponse);
        } else if (partialResponseWithContextHandler != null) {
            PartialResponseContext context = new PartialResponseContext(new CancellationUnsupportedStreamingHandle());
            partialResponseWithContextHandler.accept(new PartialResponse(partialResponse), context);
        }
    }

    @Override
    public void onPartialResponse(PartialResponse partialResponse, PartialResponseContext context) {
        // 与字符串重载保持一致：护栏开启时不立即透传，先缓存再集中下发。
        if (hasOutputGuardrails) {
            responseBuffer.add(partialResponse.text());
        } else if (partialResponseHandler != null) {
            partialResponseHandler.accept(partialResponse.text());
        } else if (partialResponseWithContextHandler != null) {
            partialResponseWithContextHandler.accept(partialResponse, context);
        }
    }

    @Override
    public void onPartialThinking(PartialThinking partialThinking) {
        if (partialThinkingHandler != null) {
            partialThinkingHandler.accept(partialThinking);
        } else if (partialThinkingWithContextHandler != null) {
            PartialThinkingContext context = new PartialThinkingContext(new CancellationUnsupportedStreamingHandle());
            partialThinkingWithContextHandler.accept(partialThinking, context);
        }
    }

    @Override
    public void onPartialThinking(PartialThinking partialThinking, PartialThinkingContext context) {
        if (partialThinkingHandler != null) {
            partialThinkingHandler.accept(partialThinking);
        } else if (partialThinkingWithContextHandler != null) {
            partialThinkingWithContextHandler.accept(partialThinking, context);
        }
    }

    @Override
    public void onPartialToolCall(PartialToolCall partialToolCall) {
        if (partialToolCallHandler != null) {
            partialToolCallHandler.accept(partialToolCall);
        } else if (partialToolCallWithContextHandler != null) {
            PartialToolCallContext context = new PartialToolCallContext(new CancellationUnsupportedStreamingHandle());
            partialToolCallWithContextHandler.accept(partialToolCall, context);
        }
    }

    @Override
    public void onPartialToolCall(PartialToolCall partialToolCall, PartialToolCallContext context) {
        if (partialToolCallHandler != null) {
            partialToolCallHandler.accept(partialToolCall);
        } else if (partialToolCallWithContextHandler != null) {
            partialToolCallWithContextHandler.accept(partialToolCall, context);
        }
    }

    @Override
    public void onCompleteToolCall(CompleteToolCall completeToolCall) {
        if (toolExecutor != null) {
            ToolExecutionRequest toolRequest = completeToolCall.toolExecutionRequest();
            var future = CompletableFuture.supplyAsync(
                    () -> {
                        ToolExecutionResult toolResult = execute(toolRequest);
                        return new ToolRequestResult(toolRequest, toolResult);
                    },
                    toolExecutor);
            toolExecutionFutures.add(future);
        }
    }

    /**
     * 触发一次调用完成事件。
     * 该事件用于统一标记一次 AI Service 调用生命周期结束，便于观测系统统计成功完成次数。
     *
     * @param result 调用最终结果，可为模型响应或其他上层封装结果
     * @param <T> 结果类型
     */
    private <T> void fireInvocationComplete(T result) {
        context.eventListenerRegistrar.fireEvent(AiServiceCompletedEvent.builder()
                .invocationContext(invocationContext)
                .result(result)
                .build());
    }

    /**
     * 触发工具执行完成事件。
     * 该事件用于记录“哪个工具被调用、输出了什么结果”，支撑链路追踪与问题排查。
     *
     * @param toolRequestResult 工具请求及执行结果
     */
    private void fireToolExecutedEvent(ToolRequestResult toolRequestResult) {
        context.eventListenerRegistrar.fireEvent(ToolExecutedEvent.builder()
                .invocationContext(invocationContext)
                .request(toolRequestResult.request())
                .resultText(toolRequestResult.result().resultText())
                .build());
    }

    /**
     * 触发模型响应接收事件。
     * 用于标记某一轮请求已经从模型侧返回完整响应，便于按轮次观测工具链路。
     *
     * @param chatResponse 本轮收到的完整响应
     */
    private void fireResponseReceivedEvent(ChatResponse chatResponse) {
        context.eventListenerRegistrar.fireEvent(AiServiceResponseReceivedEvent.builder()
                .invocationContext(invocationContext)
                .request(chatRequest)
                .response(chatResponse)
                .build());
    }

    /**
     * 触发请求发出事件。
     * 工具执行后续轮请求会再次调用该方法，保证每一轮请求都可独立追踪。
     *
     * @param chatRequest 即将发送到模型的新请求
     */
    private void fireRequestIssuedEvent(ChatRequest chatRequest) {
        context.eventListenerRegistrar.fireEvent(AiServiceRequestIssuedEvent.builder()
                .invocationContext(invocationContext)
                .request(chatRequest)
                .build());
    }

    /**
     * 触发错误事件。
     * 即便业务侧后续选择忽略或吞掉异常，也会先记录错误事件以保留诊断信息。
     *
     * @param error 捕获到的异常
     */
    private void fireErrorReceived(Throwable error) {
        context.eventListenerRegistrar.fireEvent(AiServiceErrorEvent.builder()
                .invocationContext(invocationContext)
                .error(error)
                .build());
    }

    /**
     * 处理一次流式响应完成事件。
     * 如果模型返回了工具调用请求，会先执行工具并决定是否继续向模型发起下一轮请求；
     * 否则直接组装最终响应并回调上层。
     *
     * @param chatResponse 本轮流式聚合后的完整响应
     */
    @Override
    public void onCompleteResponse(ChatResponse chatResponse) {
        // 首先上报“响应已接收”事件，保证每轮响应都可被外部监听到。
        fireResponseReceivedEvent(chatResponse);
        // 提取并写入 AI 消息到临时/持久记忆，作为后续工具调用与续轮输入。
        AiMessage aiMessage = chatResponse.aiMessage();
        addToMemory(aiMessage);

        // 只有包含工具调用请求时，才进入工具执行与续轮推理分支。
        if (aiMessage.hasToolExecutionRequests()) {

            // 限制顺序工具调用次数，防止异常提示词导致无限续轮。
            if (sequentialToolsInvocationsLeft-- == 0) {
                throw runtime(
                        "Something is wrong, exceeded %s sequential tool invocations",
                        context.toolService.maxSequentialToolsInvocations());
            }

            // 向上层回调中间响应，便于调用方观察“工具前”阶段结果。
            if (intermediateResponseHandler != null) {
                intermediateResponseHandler.accept(chatResponse);
            }

            // 仅当所有工具都声明可立即返回时，才允许跳过下一轮模型推理。
            boolean immediateToolReturn = true;

            if (toolExecutor != null) {
                // 有异步执行器时，等待并收集所有工具执行 Future。
                for (Future<ToolRequestResult> toolExecutionFuture : toolExecutionFutures) {
                    try {
                        // 取回工具执行结果并上报工具执行事件。
                        ToolRequestResult toolRequestResult = toolExecutionFuture.get();
                        fireToolExecutedEvent(toolRequestResult);
                        // 将工具结果写入记忆，供下一轮模型读取上下文。
                        ToolExecutionResultMessage toolExecutionResultMessage = ToolExecutionResultMessage.from(
                                toolRequestResult.request(),
                                toolRequestResult.result().resultText());
                        addToMemory(toolExecutionResultMessage);
                        // 只要有一个工具不是“立即返回”，就必须继续下一轮模型调用。
                        immediateToolReturn = immediateToolReturn
                                && context.toolService.isImmediateTool(toolExecutionResultMessage.toolName());
                    } catch (ExecutionException e) {
                        // 保留业务运行时异常类型，避免被包装后丢失语义。
                        if (e.getCause() instanceof RuntimeException re) {
                            throw re;
                        } else {
                            throw new RuntimeException(e.getCause());
                        }
                    } catch (InterruptedException e) {
                        // 恢复线程中断标记，避免吞掉中断信号。
                        Thread.currentThread().interrupt();
                        throw new RuntimeException(e);
                    }
                }
            } else {
                // 无异步执行器时按顺序同步执行所有工具。
                for (ToolExecutionRequest toolRequest : aiMessage.toolExecutionRequests()) {
                    ToolExecutionResult toolResult = execute(toolRequest);
                    ToolRequestResult toolRequestResult = new ToolRequestResult(toolRequest, toolResult);
                    // 同步模式下同样要发送工具执行事件并写入记忆。
                    fireToolExecutedEvent(toolRequestResult);
                    addToMemory(ToolExecutionResultMessage.from(toolRequest, toolResult.resultText()));
                    // 与异步分支一致，累计是否可立即返回。
                    immediateToolReturn =
                            immediateToolReturn && context.toolService.isImmediateTool(toolRequest.name());
                }
            }

            // 全部工具都允许立即返回时，直接产出最终响应，结束本次流。
            if (immediateToolReturn) {
                ChatResponse finalChatResponse = finalResponse(chatResponse, aiMessage);
                fireInvocationComplete(finalChatResponse);

                // 有完成回调时通知调用方最终结果。
                if (completeResponseHandler != null) {
                    completeResponseHandler.accept(finalChatResponse);
                }
                return;
            }

            // 工具执行后需要把新记忆再次发送给模型，触发下一轮推理。
            ChatRequest nextChatRequest = context.chatRequestTransformer.apply(
                    ChatRequest.builder()
                            .messages(messagesToSend(invocationContext.chatMemoryId()))
                            .parameters(chatRequestParameters(invocationContext.methodArguments(), toolSpecifications))
                            .build(),
                    invocationContext.chatMemoryId());

            var handler = new AiServiceStreamingResponseHandler(
                    nextChatRequest,
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
                    temporaryMemory,
                    TokenUsage.sum(tokenUsage, chatResponse.metadata().tokenUsage()),
                    toolSpecifications,
                    toolExecutors,
                    sequentialToolsInvocationsLeft,
                    toolArgumentsErrorHandler,
                    toolExecutionErrorHandler,
                    toolExecutor,
                    commonGuardrailParams,
                    methodKey);

            // 工具结果需要回灌给模型继续推理，因此这里递归触发下一轮流式请求。
            fireRequestIssuedEvent(nextChatRequest);
            context.streamingChatModel.chat(nextChatRequest, handler);
        } else {
            // 没有工具调用时，当前响应即最终响应。
            ChatResponse finalChatResponse = finalResponse(chatResponse, aiMessage);

            if (completeResponseHandler != null) {
                // 执行输出护栏。
                if (hasOutputGuardrails) {
                    if (commonGuardrailParams != null) {
                        // 输出护栏执行前把最新记忆注入公共参数，确保规则可看到最新上下文。
                        var newCommonParams = commonGuardrailParams.toBuilder()
                                .chatMemory(getMemory())
                                .build();

                        var outputGuardrailParams = OutputGuardrailRequest.builder()
                                .responseFromLLM(finalChatResponse)
                                .chatExecutor(chatExecutor)
                                .requestParams(newCommonParams)
                                .build();

                        finalChatResponse =
                                context.guardrailService().executeGuardrails(methodKey, outputGuardrailParams);
                    }

                    // 启用输出护栏时，需要先完成护栏处理，再统一回放缓存的 partial 响应。
                    if (partialResponseHandler != null) {
                        // 护栏场景下延迟下发 partial，这里统一冲刷缓存，确保最终输出一致。
                        responseBuffer.forEach(partialResponseHandler::accept);
                    }
                    // partial 已下发后清空缓存，避免重复回调。
                    responseBuffer.clear();
                }

                // 先发完成事件，再回调调用方完成处理器，保持事件顺序稳定。
                fireInvocationComplete(finalChatResponse);
                completeResponseHandler.accept(finalChatResponse);
            } else {
                // 即使调用方未注册完成回调，也要发完成事件保证链路闭环。
                fireInvocationComplete(finalChatResponse);
            }
        }
    }

    /**
     * 组装最终响应对象并累计 token 用量。
     * 多轮工具调用场景下，历史轮次 token 会先保存在 {@code tokenUsage} 中，
     * 这里与当前轮 token 叠加后写入最终元数据，避免统计口径丢失。
     *
     * @param completeResponse 当前轮完整响应
     * @param aiMessage 最终确认要返回的 AI 消息
     * @return 合并 token 用量后的最终响应
     */
    private ChatResponse finalResponse(ChatResponse completeResponse, AiMessage aiMessage) {
        return ChatResponse.builder()
                .aiMessage(aiMessage)
                .metadata(completeResponse.metadata().toBuilder()
                        .tokenUsage(tokenUsage.add(completeResponse.metadata().tokenUsage()))
                        .build())
                .build();
    }

    /**
     * 执行单个工具请求。
     * 统一通过 {@link dev.langchain4j.service.tool.ToolService} 执行，
     * 以复用参数校验、回调触发与错误处理策略。
     *
     * @param toolRequest 工具执行请求
     * @return 工具执行结果
     */
    private ToolExecutionResult execute(ToolExecutionRequest toolRequest) {
        return context.toolService.executeTool(
                invocationContext, toolExecutors, toolRequest, beforeToolExecutionHandler, toolExecutionHandler);
    }

    /**
     * 获取当前调用对应的记忆实例。
     * 默认使用当前调用绑定的 memoryId，确保同一请求链路访问同一份上下文。
     *
     * @return 当前调用可写记忆
     */
    private ChatMemory getMemory() {
        return getMemory(invocationContext.chatMemoryId());
    }

    /**
     * 根据给定 memoryId 获取记忆实例。
     * 有外部记忆服务时复用持久记忆；否则回退到本次流式调用内部的临时记忆。
     *
     * @param memId 目标 memoryId（用于表达调用意图）
     * @return 对应的记忆实例
     */
    private ChatMemory getMemory(Object memId) {
        return context.hasChatMemory()
                ? context.chatMemoryService.getOrCreateChatMemory(invocationContext.chatMemoryId())
                : temporaryMemory;
    }

    /**
     * 将消息写入当前记忆。
     * 所有 AI 响应与工具结果都会先入记忆，保证续轮请求看到完整上下文。
     *
     * @param chatMessage 需要写入的消息
     */
    private void addToMemory(ChatMessage chatMessage) {
        getMemory().add(chatMessage);
    }

    /**
     * 获取下一轮发送给模型的消息列表。
     * 该方法从记忆中读取最新上下文，确保工具执行结果已被包含在续轮输入中。
     *
     * @param memoryId 当前轮请求使用的 memoryId
     * @return 发往模型的消息列表
     */
    private List<ChatMessage> messagesToSend(Object memoryId) {
        return getMemory(memoryId).messages();
    }

    /**
     * 处理流式链路中的异常。
     * 若业务方提供了错误回调，则先上报错误事件再交由业务处理；
     * 若未提供，则记录告警日志并按“忽略错误”策略收口。
     *
     * @param error 当前链路捕获到的异常
     */
    @Override
    public void onError(Throwable error) {
        if (errorHandler != null) {
            try {
                // 先记录统一错误事件，保证观测系统与业务回调看到同一异常事实。
                fireErrorReceived(error);
                // 再将错误交由业务回调处理，允许上层决定重试/降级策略。
                errorHandler.accept(error);
            } catch (Exception e) {
                // 业务错误回调再次抛错时，保留原始异常与回调异常双份日志便于定位。
                LOG.error("While handling the following error...", error);
                LOG.error("...the following error happened", e);
            }
        } else {
            // 未配置错误回调时仅记录告警，保持流式链路在“忽略错误”模式下可继续运行。
            LOG.warn("Ignored error", error);
        }
    }
}
