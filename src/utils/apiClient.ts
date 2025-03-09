import { ChatCompletionRequest, ChatCompletionResponse } from '@/types/api';
import { ChatMessage } from '@/types/chat';
import { CHAT_SETTINGS } from '@/config/chat';
import { logger } from '@/utils/logger';

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
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxRetries: 3,
  initialDelay: 1000,
  maxDelay: 10000,
  backoffFactor: 2,
};

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any,
    public requestId?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class ApiClient {
  private static instance: ApiClient;
  private retryOptions: RetryOptions;
  private readonly baseUrl = '/api';
  private metrics: ApiMetrics = {
    requestCount: 0,
    errorCount: 0,
    retryCount: 0,
    averageResponseTime: 0,
    totalResponseTime: 0,
  };

  private constructor(options: Partial<RetryOptions> = {}) {
    this.retryOptions = { ...DEFAULT_RETRY_OPTIONS, ...options };
  }

  static getInstance(options?: Partial<RetryOptions>): ApiClient {
    if (!ApiClient.instance) {
      ApiClient.instance = new ApiClient(options);
    }
    return ApiClient.instance;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private async delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private getRetryDelay(attempt: number): number {
    const delay = this.retryOptions.initialDelay * 
      Math.pow(this.retryOptions.backoffFactor, attempt);
    return Math.min(delay, this.retryOptions.maxDelay);
  }

  private isRetryableError(error: any): boolean {
    if (error instanceof ApiError) {
      // Retry on network errors and specific HTTP status codes
      return error.status >= 500 || error.status === 429;
    }
    return true; // Retry on network/unknown errors
  }

  private validateResponse(data: any): data is ChatCompletionResponse {
    logger.debug('Validating API response', {
      hasData: !!data,
      hasChoices: data && Array.isArray(data.choices),
      choicesLength: data?.choices?.length,
      hasMessage: data?.choices?.[0]?.message,
      messageContent: typeof data?.choices?.[0]?.message?.content,
      messageRole: typeof data?.choices?.[0]?.message?.role,
      rawResponse: data
    });

    return (
      data &&
      Array.isArray(data.choices) &&
      data.choices.length > 0 &&
      data.choices[0].message &&
      typeof data.choices[0].message.content === 'string' &&
      typeof data.choices[0].message.role === 'string'
    );
  }

  private updateMetrics(startTime: number, isError: boolean = false, retries: number = 0) {
    const responseTime = Date.now() - startTime;
    this.metrics.requestCount++;
    this.metrics.totalResponseTime += responseTime;
    this.metrics.averageResponseTime = this.metrics.totalResponseTime / this.metrics.requestCount;
    
    if (isError) this.metrics.errorCount++;
    if (retries > 0) this.metrics.retryCount += retries;

    // Log metrics if they exceed thresholds
    if (this.metrics.averageResponseTime > 2000 || this.metrics.errorCount / this.metrics.requestCount > 0.1) {
      console.warn('API Metrics Warning:', {
        ...this.metrics,
        errorRate: this.metrics.errorCount / this.metrics.requestCount,
      });
    }
  }

  async fetchWithRetry<T>(
    url: string,
    options: RequestInit,
    customRetryOptions?: Partial<RetryOptions>
  ): Promise<T> {
    const startTime = Date.now();
    const requestId = this.generateRequestId();
    const retryOpts = { ...this.retryOptions, ...customRetryOptions };
    let lastError: Error | null = null;
    let retryCount = 0;

    // Add request ID to headers
    options.headers = {
      ...options.headers,
      'X-Request-ID': requestId,
    };

    logger.info(`[${requestId}] Starting request to ${url}`, {
      method: options.method,
      headers: options.headers,
      bodyLength: options.body ? (options.body as string).length : 0
    });

    for (let attempt = 0; attempt <= retryOpts.maxRetries; attempt++) {
      try {
        if (attempt > 0) {
          retryCount++;
          const delay = this.getRetryDelay(attempt - 1);
          await this.delay(delay);
          logger.info(`[${requestId}] Retry attempt ${attempt} after ${delay}ms`);
        }

        const response = await fetch(url, options);
        const data = await response.json();

        logger.debug(`[${requestId}] Received response`, {
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers.entries()),
          data
        });

        if (!response.ok) {
          throw new ApiError(
            data.error?.message || `API request failed with status ${response.status}`,
            response.status,
            data,
            requestId
          );
        }

        // Validate response structure
        if (!this.validateResponse(data)) {
          logger.error(`[${requestId}] Invalid response format`, {
            data,
            validationErrors: this.getValidationErrors(data)
          });
          throw new ApiError(
            'Invalid response format from API',
            500,
            data,
            requestId
          );
        }

        logger.info(`[${requestId}] Request completed successfully`);
        this.updateMetrics(startTime, false, retryCount);
        return data as T;
      } catch (error) {
        logger.error(`[${requestId}] API error (attempt ${attempt + 1}/${retryOpts.maxRetries + 1}):`, {
          error,
          errorMessage: error instanceof Error ? error.message : String(error),
          errorStack: error instanceof Error ? error.stack : undefined,
          attempt,
          maxRetries: retryOpts.maxRetries
        });
        
        lastError = error as Error;
        
        if (!this.isRetryableError(error) || attempt === retryOpts.maxRetries) {
          this.updateMetrics(startTime, true, retryCount);
          throw error;
        }
      }
    }

    this.updateMetrics(startTime, true, retryCount);
    throw lastError;
  }

  private getValidationErrors(data: any): string[] {
    const errors: string[] = [];
    if (!data) errors.push('Response data is null or undefined');
    if (!Array.isArray(data?.choices)) errors.push('choices is not an array');
    if (!data?.choices?.length) errors.push('choices array is empty');
    if (!data?.choices?.[0]?.message) errors.push('first choice has no message');
    if (typeof data?.choices?.[0]?.message?.content !== 'string') errors.push('message content is not a string');
    if (typeof data?.choices?.[0]?.message?.role !== 'string') errors.push('message role is not a string');
    return errors;
  }

  async sendChatRequest(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    return this.fetchWithRetry<ChatCompletionResponse>(
      '/api/chat',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      }
    );
  }

  async sendChatMessage(
    messages: ChatMessage[], 
    image?: string | null, 
    promptStyle?: string, 
    knowledgeFocus?: string, 
    citeSources?: boolean,
    agentic?: boolean
  ): Promise<ChatCompletionResponse> {
    const requestId = this.generateRequestId();
    console.log(`[${requestId}] Sending chat message with ${messages.length} messages${image ? ' and image' : ''}${promptStyle ? `, prompt style: ${promptStyle}` : ''}${knowledgeFocus ? `, knowledge focus: ${knowledgeFocus}` : ''}${citeSources !== undefined ? `, cite sources: ${citeSources}` : ''}${agentic ? ', using multi-agent mode' : ''}`);

    try {
      // If agentic mode is enabled, route to the multi-agent backend
      if (agentic) {
        // Extract just the text content from the most recent user message
        const userMessage = messages.filter(msg => msg.role === 'user').pop()?.content || '';
        
        // Make the request to the multi-agent API
        return await this.sendMultiAgentRequest(userMessage, image);
      }
      
      // Normal LLM mode - continue with existing code
      return await this.fetchWithRetry<ChatCompletionResponse>(
        `${this.baseUrl}/chat`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages,
            image,
            promptStyle,
            knowledgeFocus,
            citeSources
          }),
        }
      );
    } catch (error) {
      logger.error(`[${requestId}] Error sending chat message:`, error);
      throw error;
    }
  }

  // New method to handle multi-agent API requests
  private async sendMultiAgentRequest(
    query: string,
    image?: string | null
  ): Promise<ChatCompletionResponse> {
    const requestId = this.generateRequestId();
    logger.info(`[${requestId}] Sending request to multi-agent API`, { query });
    
    try {
      // Make a request to the multi-agent backend
      const response = await this.fetchWithRetry<any>(
        // Point to the multi-agent backend endpoint
        'http://localhost:8000/api/v1/chat',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query,
            image
          }),
        }
      );
      
      // If the response from the multi-agent system returns a task_id, we need to poll for results
      if (response.task_id) {
        logger.info(`[${requestId}] Multi-agent task started, polling for results`, { taskId: response.task_id });
        
        // For now, just return a message indicating the task has started
        // In a complete implementation, we would poll the /api/v1/chat/{task_id} endpoint
        return {
          choices: [{
            message: {
              content: `Multi-agent task started. Task ID: ${response.task_id}. The system is processing your request, which may take some time.`,
              role: 'assistant'
            },
            finish_reason: 'function_call'
          }],
          usage: {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0
          }
        };
      }
      
      // Format the response to match ChatCompletionResponse
      return {
        choices: [{
          message: {
            content: response.result || response.message || 'No response from multi-agent system',
            role: 'assistant'
          },
          finish_reason: 'stop'
        }],
        usage: response.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
    } catch (error) {
      logger.error(`[${requestId}] Error sending multi-agent request:`, error);
      
      // Return a user-friendly error message
      return {
        choices: [{
          message: {
            content: 'There was an error connecting to the multi-agent system. Please ensure the backend is running and try again.',
            role: 'assistant'
          },
          finish_reason: 'error'
        }],
        usage: {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
    }
  }

  // Method to get current metrics
  getMetrics(): ApiMetrics {
    return { ...this.metrics };
  }

  // Method to reset metrics
  resetMetrics(): void {
    this.metrics = {
      requestCount: 0,
      errorCount: 0,
      retryCount: 0,
      averageResponseTime: 0,
      totalResponseTime: 0,
    };
  }
}

export const apiClient = ApiClient.getInstance(); 