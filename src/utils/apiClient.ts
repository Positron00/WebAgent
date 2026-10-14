import { ChatCompletionRequest, ChatCompletionResponse } from "@/types/api";
import { ChatMessage } from "@/types/chat";
import { AgenticMode, RouteInfo } from "@/types/routing";
import { CHAT_SETTINGS } from "@/config/chat";
import { logger } from "@/utils/logger";
import { clientConfig } from "@/config/clientConfig";
import { routeQuestion } from "@/utils/routeQuestion";
import { isGuardRefusal, normalizeSecuritySummary } from "@/types/security";
import { RESEARCH_TIMEOUT_MESSAGE, ResearchFailed } from "@/types/research";
import { fetchWithRetry } from "@/utils/fetchWithRetry";
import { isSignInRequired, redirectToSignIn } from "@/utils/signIn";

const TRIAGE_ROUTING_FALLBACK_NOTICE =
  "I'm answering directly because research routing is temporarily unavailable.\n\n";

export function buildPreviousMessages(
  history: ChatMessage[],
): { role: "user" | "assistant"; content: string }[] {
  return history
    .filter(
      (m) =>
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string" &&
        m.content.trim(),
    )
    .slice(-8)
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content.slice(0, 4000),
    }));
}

interface RetryOptions {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  backoffFactor: number;
}

interface ApiMetrics {
  requestCount: number;
  errorCount: number;
  retryCount: number;
  averageResponseTime: number;
  totalResponseTime: number;
  lastRequestTime: number | null;
  consecutiveErrors: number;
  statusCodes: Record<string, number>;
}

interface ChatCompletionMessage {
  role: string;
  content: string;
  logs?: string[];
}

/** Next.js routes return `{ error: string }`; some APIs use `{ error: { message } }`. */
function extractErrorMessage(errorData: unknown): string | undefined {
  if (!errorData || typeof errorData !== "object") {
    return undefined;
  }
  const err = (errorData as { error?: unknown; message?: unknown }).error;
  if (typeof err === "string" && err.trim()) {
    return err;
  }
  if (err && typeof err === "object" && "message" in err) {
    const message = (err as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }
  const topLevel = (errorData as { message?: unknown }).message;
  if (typeof topLevel === "string" && topLevel.trim()) {
    return topLevel;
  }
  return undefined;
}

/** Errors that end a research request as they are: no further polling and no fallback to /api/chat. */
function isFinalResearchError(error: unknown): boolean {
  if (error instanceof ResearchFailed) return true;
  return (
    error instanceof ApiError &&
    (isGuardRefusal(error.data) || isSignInRequired(error.status, error.data))
  );
}

/**
 * Enhanced error class with additional context for better debugging
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any,
    public requestId?: string,
    public endpoint?: string,
    public timestamp: number = Date.now(),
  ) {
    super(message);
    this.name = "ApiError";

    // Capture stack trace for better debugging
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }

    // Log the error with context
    logger.error("API Error", {
      message,
      status,
      requestId,
      endpoint,
      timestamp: new Date(timestamp).toISOString(),
      data: typeof data === "object" ? JSON.stringify(data) : data,
    });
  }
}

/**
 * API Client for handling communication with backend services
 * Enhanced with better error handling, metrics, and observability
 */
export class ApiClient {
  private static instance: ApiClient;
  private retryOptions: RetryOptions;
  private readonly baseUrl = clientConfig.api.baseUrl;
  private metrics: ApiMetrics = {
    requestCount: 0,
    errorCount: 0,
    retryCount: 0,
    averageResponseTime: 0,
    totalResponseTime: 0,
    lastRequestTime: null,
    consecutiveErrors: 0,
    statusCodes: {},
  };

  private constructor(options: Partial<RetryOptions> = {}) {
    this.retryOptions = {
      maxRetries: options.maxRetries ?? 3,
      initialDelay: options.initialDelay ?? 500,
      maxDelay: options.maxDelay ?? 5000,
      backoffFactor: options.backoffFactor ?? 2,
    };

    logger.info("ApiClient initialized", {
      retryOptions: this.retryOptions,
      baseUrl: this.baseUrl,
      environment: clientConfig.environment,
    });
  }

  static getInstance(options?: Partial<RetryOptions>): ApiClient {
    if (!ApiClient.instance) {
      ApiClient.instance = new ApiClient(options);
    }
    return ApiClient.instance;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
  }

  private async delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private getRetryDelay(attempt: number): number {
    const delay = Math.min(
      this.retryOptions.initialDelay *
        Math.pow(this.retryOptions.backoffFactor, attempt),
      this.retryOptions.maxDelay,
    );
    // Add jitter to prevent synchronized retries
    return delay * (0.8 + Math.random() * 0.4);
  }

  private isRetryableError(error: any): boolean {
    // Network errors should be retried
    if (error instanceof TypeError && error.message.includes("network")) {
      return true;
    }

    // Retry server errors (5xx) but not client errors (4xx)
    if (error instanceof ApiError) {
      return error.status >= 500 && error.status < 600;
    }

    // Timeout errors should be retried
    if (error instanceof Error && error.message.includes("timeout")) {
      return true;
    }

    return false;
  }

  private validateResponse(data: any): data is ChatCompletionResponse {
    const errors = this.getValidationErrors(data);

    if (errors.length > 0) {
      logger.warn("Invalid API response", {
        errors,
        data: JSON.stringify(data).substring(0, 500),
      });
      return false;
    }

    return true;
  }

  private updateMetrics(
    startTime: number,
    isError: boolean = false,
    retries: number = 0,
    statusCode?: number,
  ) {
    const requestTime = Date.now() - startTime;

    this.metrics.requestCount++;
    this.metrics.totalResponseTime += requestTime;
    this.metrics.averageResponseTime =
      this.metrics.totalResponseTime / this.metrics.requestCount;
    this.metrics.lastRequestTime = Date.now();

    if (statusCode) {
      const statusCodeKey = `${statusCode}`;
      this.metrics.statusCodes[statusCodeKey] =
        (this.metrics.statusCodes[statusCodeKey] || 0) + 1;
    }

    if (isError) {
      this.metrics.errorCount++;
      this.metrics.consecutiveErrors++;
    } else {
      this.metrics.consecutiveErrors = 0;
    }

    if (retries > 0) {
      this.metrics.retryCount += retries;
    }

    // Log metrics periodically (every 10 requests)
    if (this.metrics.requestCount % 10 === 0) {
      logger.info("API Metrics", {
        ...this.metrics,
        errorRate: this.metrics.errorCount / this.metrics.requestCount,
        retryRate: this.metrics.retryCount / this.metrics.requestCount,
      });
    }
  }

  async fetchWithRetry<T>(
    url: string,
    options: RequestInit,
    customRetryOptions?: Partial<RetryOptions>,
  ): Promise<T> {
    const retryOptions: RetryOptions = {
      ...this.retryOptions,
      ...customRetryOptions,
    };

    const requestId = this.generateRequestId();
    const startTime = Date.now();
    const fullUrl = url.startsWith("http") ? url : `${this.baseUrl}${url}`;

    // Add request ID to headers for tracing
    const headers = new Headers(options.headers || {});
    headers.set("X-Request-ID", requestId);
    const { signal, ...init } = options;

    logger.debug("API request", {
      method: options.method,
      url: fullUrl,
      requestId,
      attempt: 0,
    });

    return fetchWithRetry<T>(
      fullUrl,
      { ...init, headers },
      {
        timeoutMs: clientConfig.api.timeoutMs,
        maxRetries: retryOptions.maxRetries,
        signal: signal ?? undefined,
        isRetryable: (error) => this.isRetryableError(error),
        retryDelay: (attempt) => this.getRetryDelay(attempt),
        delay: (ms) => this.delay(ms),
        readResponse: async (response, attempt) => {
          // Update metrics with status code
          this.updateMetrics(startTime, !response.ok, attempt, response.status);

          if (!response.ok) {
            let errorData;
            try {
              errorData = await response.json();
            } catch (e) {
              errorData = { error: response.statusText };
            }

            if (isSignInRequired(response.status, errorData)) {
              redirectToSignIn();
              throw new ApiError(
                "Sign-in required",
                401,
                errorData,
                requestId,
                fullUrl,
              );
            }

            const apiMessage = extractErrorMessage(errorData);
            throw new ApiError(
              apiMessage || `HTTP Error: ${response.statusText}`,
              response.status,
              errorData,
              requestId,
              fullUrl,
            );
          }

          const data = await response.json();

          // For chat completions, validate the response structure
          if (url.includes("/chat") && !this.validateResponse(data)) {
            throw new ApiError(
              "Invalid API response format",
              200,
              data,
              requestId,
              fullUrl,
            );
          }

          logger.debug("API response", {
            method: options.method,
            url: fullUrl,
            requestId,
            statusCode: response.status,
            duration: Date.now() - startTime,
            attempt,
          });

          return data;
        },
        onError: (error: any, attempt, canRetry, delayMs) => {
          logger.error("API request failed", {
            method: options.method,
            url: fullUrl,
            requestId,
            attempt,
            error: error?.message,
            status: error instanceof ApiError ? error.status : undefined,
            isRetryable: this.isRetryableError(error),
            canRetry,
            retryCount: attempt,
            maxRetries: retryOptions.maxRetries,
          });

          if (!canRetry) {
            this.updateMetrics(startTime, true, attempt);
            return;
          }

          logger.info("Retrying API request", {
            method: options.method,
            url: fullUrl,
            requestId,
            attempt: attempt + 1,
            delayMs,
          });
        },
      },
    );
  }

  private getValidationErrors(data: any): string[] {
    const errors = [];

    if (typeof data !== "object" || data === null)
      errors.push("response is not an object");
    if (!Array.isArray(data?.choices)) errors.push("choices is not an array");
    if (data?.choices?.length === 0) errors.push("choices array is empty");
    if (typeof data?.choices?.[0] !== "object")
      errors.push("first choice is not an object");
    if (typeof data?.choices?.[0]?.message !== "object")
      errors.push("message is not an object");
    if (typeof data?.choices?.[0]?.message?.content !== "string")
      errors.push("message content is not a string");
    if (typeof data?.choices?.[0]?.message?.role !== "string")
      errors.push("message role is not a string");
    return errors;
  }

  async sendChatRequest(
    request: ChatCompletionRequest,
  ): Promise<ChatCompletionResponse> {
    return this.fetchWithRetry<ChatCompletionResponse>(
      clientConfig.api.endpoints.chat,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      },
    );
  }

  async sendChatMessage(
    messages: ChatMessage[],
    image?: string | null,
    promptStyle?: string,
    knowledgeFocus?: string,
    citeSources?: boolean,
    mode: AgenticMode = "off",
    options?: {
      forceAgentic?: boolean;
      consentToken?: string;
      approvalTokens?: string[];
    },
  ): Promise<ChatCompletionResponse> {
    const history = messages.slice(0, -1);
    const lastUserMessage = [...messages]
      .reverse()
      .find((m) => m.role === "user");
    const query =
      typeof lastUserMessage?.content === "string"
        ? lastUserMessage.content.trim()
        : "";

    if (!query && !image) {
      throw new ApiError("No user message to send", 400);
    }

    const previousUserQuestion = (() => {
      for (let i = history.length - 1; i >= 0; i--) {
        if (
          history[i].role === "user" &&
          typeof history[i].content === "string"
        ) {
          return history[i].content;
        }
      }
      return null;
    })();

    let routeInfo: RouteInfo;
    let fallbackNotice = false;

    if (options?.forceAgentic) {
      routeInfo = {
        route: "agentic",
        decidedBy: "override",
        reason: "You asked for research",
      };
    } else {
      const outcome = await routeQuestion({
        mode,
        query,
        previousUserQuestion,
        hasImage: !!image,
      });

      if (outcome.kind === "rate_limited") {
        throw new ApiError("Rate limit exceeded. Please try again later.", 429);
      }

      routeInfo = outcome.info;
      fallbackNotice = outcome.fallbackNotice ?? false;
    }

    // An image with no text (Always mode) gives the research agents nothing to work on,
    // and they can't see images; the direct model can.
    if (routeInfo.route === "agentic" && !query) {
      routeInfo = {
        route: "direct",
        decidedBy: "rule",
        reason: "Image without a question",
      };
    }

    let response: ChatCompletionResponse;

    if (routeInfo.route === "direct") {
      logger.info("Routing to direct LLM", {
        mode,
        decidedBy: routeInfo.decidedBy,
      });
      response = await this.sendDirectChatRequest(
        messages,
        image,
        promptStyle,
        knowledgeFocus,
        citeSources,
        fallbackNotice,
        options?.consentToken,
      );
    } else {
      logger.info("Routing to multi-agent system", {
        mode,
        decidedBy: routeInfo.decidedBy,
      });
      const agenticResult = await this.sendMultiAgentRequest(
        query,
        image,
        buildPreviousMessages(history),
        options?.forceAgentic,
        options?.consentToken,
        options?.approvalTokens,
      );
      response = agenticResult.response;
      if (agenticResult.usedFallback) {
        routeInfo = {
          route: "direct",
          decidedBy: "fallback",
          reason: "Research agents unavailable",
        };
      }
    }

    return { ...response, route: routeInfo };
  }

  private async sendDirectChatRequest(
    messages: ChatMessage[],
    image?: string | null,
    promptStyle?: string,
    knowledgeFocus?: string,
    citeSources?: boolean,
    fallbackNotice?: boolean,
    consentToken?: string,
  ): Promise<ChatCompletionResponse> {
    const requestMessages = messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));

    const response = await this.fetchWithRetry<ChatCompletionResponse>(
      clientConfig.api.endpoints.chat,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: requestMessages,
          model: clientConfig.together.model,
          max_tokens: CHAT_SETTINGS.maxTokens,
          temperature: CHAT_SETTINGS.temperature,
          top_p: CHAT_SETTINGS.topP,
          frequency_penalty: CHAT_SETTINGS.frequencyPenalty,
          presence_penalty: CHAT_SETTINGS.presencePenalty,
          promptStyle,
          knowledgeFocus,
          citeSources,
          agentic: false,
          image: image || null,
          ...(consentToken ? { consentToken } : {}),
        }),
      },
    );

    if (fallbackNotice && response.choices[0]?.message?.content) {
      response.choices[0].message.content =
        TRIAGE_ROUTING_FALLBACK_NOTICE + response.choices[0].message.content;
    }

    return response;
  }

  // Method to handle multi-agent API requests
  private async sendMultiAgentRequest(
    query: string,
    image?: string | null,
    previousMessages: { role: "user" | "assistant"; content: string }[] = [],
    triageOverride?: boolean,
    consentToken?: string,
    approvalTokens?: string[],
  ): Promise<{ response: ChatCompletionResponse; usedFallback: boolean }> {
    const requestId = this.generateRequestId();
    const startTime = Date.now();

    try {
      logger.info("Sending multi-agent request", { requestId, query });

      // Prepare the request payload
      const requestBody = {
        query,
        image: image || undefined,
        agentic: true,
        previous_messages: previousMessages,
        ...(triageOverride ? { triage_override: true } : {}),
        ...(consentToken ? { consent_token: consentToken } : {}),
        ...(approvalTokens?.length ? { approval_tokens: approvalTokens } : {}),
      };

      // Use the multi-agent API endpoint
      const response = await this.fetchWithRetry<{
        task_id?: string;
        status?: string;
      }>(clientConfig.api.endpoints.task, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Request-ID": requestId,
        },
        body: JSON.stringify(requestBody),
      });

      if (response.task_id) {
        const polled = await this.pollTaskResult(response.task_id);
        return { response: polled, usedFallback: false };
      }

      throw new Error("Multi-agent API did not return a task_id");
    } catch (error) {
      if (isFinalResearchError(error)) {
        throw error;
      }
      logger.error(
        "Multi-agent request failed, falling back to Together AI LLM",
        {
          requestId,
          error: error instanceof Error ? error.message : String(error),
          query: query.substring(0, 100), // Only log the beginning of the query for privacy
          fallback: "togetherlm",
        },
      );

      // Fallback to Together AI LLM (with agentic=false)
      try {
        logger.info(
          "Using Together AI fallback for failed multi-agent request",
          { requestId },
        );

        // Create a simple message array with just the user query
        const fallbackMessages = [
          {
            role: "system",
            content:
              "You are a helpful assistant. The multi-agent system is currently unavailable, so you are operating in fallback mode.",
          },
          {
            role: "user",
            content: query,
          },
        ];

        // Send request to Together AI
        const fallbackResponse =
          await this.fetchWithRetry<ChatCompletionResponse>(
            clientConfig.api.endpoints.chat,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-Request-ID": `${requestId}_fallback`,
              },
              body: JSON.stringify({
                messages: fallbackMessages,
                model: clientConfig.together.model,
                max_tokens: CHAT_SETTINGS.maxTokens,
                temperature: CHAT_SETTINGS.temperature,
                top_p: CHAT_SETTINGS.topP,
                agentic: false, // Explicitly mark as non-agentic for the fallback
                image: image || null,
                requestType: "multi-agent-fallback", // Indicate this is a fallback request
              }),
            },
          );

        return { response: fallbackResponse, usedFallback: true };
      } catch (fallbackError) {
        logger.error("Fallback to Together AI also failed", {
          requestId,
          originalError: error instanceof Error ? error.message : String(error),
          fallbackError:
            fallbackError instanceof Error
              ? fallbackError.message
              : String(fallbackError),
        });

        // Create a generic error response for better user experience
        return {
          response: {
            choices: [
              {
                message: {
                  role: "assistant",
                  content:
                    "I apologize, but I encountered an error processing your request. Both the multi-agent system and the fallback service are currently unavailable. Please try again later.",
                },
                finish_reason: "stop",
              },
            ],
            usage: {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
            },
          },
          usedFallback: true,
        };
      }
    }
  }

  /**
   * Poll for task results when using the async multi-agent system
   */
  private async pollTaskResult(
    taskId: string,
  ): Promise<ChatCompletionResponse> {
    const deadlineMs = 8 * 60 * 1000;
    const pollInterval = 1000;
    const maxPollInterval = 5000;
    const pollBackoffFactor = 1.5;

    const started = Date.now();
    let pollCount = 0;
    let currentInterval = pollInterval;
    let failureMessage: string | null = null;

    while (Date.now() - started < deadlineMs) {
      await this.delay(currentInterval);
      pollCount++;

      try {
        const result = await this.fetchWithRetry<any>(
          `${clientConfig.api.endpoints.task}/${taskId}`,
          { method: "GET" },
        );

        if (result.status === "completed" && result.result) {
          const security = normalizeSecuritySummary(
            result.security ?? result.result?.security,
          );
          return {
            choices: [
              {
                message: {
                  role: "assistant",
                  content:
                    result.result.content || JSON.stringify(result.result),
                  ...(result.result.logs && { logs: result.result.logs }),
                },
                finish_reason: "stop",
              },
            ],
            usage: result.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
            },
            ...(security ? { security } : {}),
          };
        }

        if (result.status === "failed") {
          if (result.error_code === "escalated") {
            throw new ApiError(
              result.error || "Task escalated for review",
              403,
              {
                code: "escalated",
                reason: result.error || "Task escalated for review",
              },
              undefined,
              `${clientConfig.api.endpoints.task}/${taskId}`,
            );
          }
          if (result.error_code === "timeout") {
            throw new ResearchFailed("timeout", RESEARCH_TIMEOUT_MESSAGE);
          }
          failureMessage = result.error || "Unknown error";
          break;
        }

        // Increase polling interval with backoff
        currentInterval = Math.min(
          currentInterval * pollBackoffFactor,
          maxPollInterval,
        );

        logger.info("Polling task", {
          taskId,
          pollCount,
          status: result.status,
          nextPollIn: currentInterval,
        });
      } catch (error) {
        if (isFinalResearchError(error)) {
          throw error;
        }
        logger.error("Error polling task", {
          taskId,
          pollCount,
          error: error instanceof Error ? error.message : String(error),
        });

        // Increase polling interval on error
        currentInterval = Math.min(currentInterval * 2, maxPollInterval);
      }
    }

    // If we've reached the maximum polling attempts, return a timeout error
    if (failureMessage) {
      throw new Error(`Task failed: ${failureMessage}`);
    }
    throw new Error(
      "Task timed out while waiting for the workflow to complete",
    );
  }

  /**
   * Get current API metrics for monitoring
   */
  getMetrics(): ApiMetrics {
    return { ...this.metrics };
  }

  /**
   * Reset metrics counters for testing
   */
  resetMetrics(): void {
    this.metrics = {
      requestCount: 0,
      errorCount: 0,
      retryCount: 0,
      averageResponseTime: 0,
      totalResponseTime: 0,
      lastRequestTime: null,
      consecutiveErrors: 0,
      statusCodes: {},
    };
  }

  /**
   * Check if the API seems to be healthy based on recent metrics
   */
  isHealthy(): boolean {
    // Consider unhealthy if there have been 3+ consecutive errors
    if (this.metrics.consecutiveErrors >= 3) {
      return false;
    }

    // Consider unhealthy if error rate is above 50%
    if (
      this.metrics.requestCount > 5 &&
      this.metrics.errorCount / this.metrics.requestCount > 0.5
    ) {
      return false;
    }

    return true;
  }
}

export const apiClient = ApiClient.getInstance();
