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
    agentic: boolean = true
  ): Promise<ChatCompletionResponse> {
    // Always use the multi-agent system for better results
    if (agentic) {
      // Get the last user message
      const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
      if (!lastUserMessage) {
        throw new Error('No user message found in the conversation');
      }
      
      return this.sendMultiAgentRequest(lastUserMessage.content, image);
    }
    
    // Legacy non-agentic path (fallback)
    const requestMessages = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));
    
    // Add system message if needed
    if (promptStyle || knowledgeFocus) {
      let systemContent = "You are a helpful assistant.";
      
      if (promptStyle) {
        systemContent += ` ${promptStyle}`;
      }
      
      if (knowledgeFocus) {
        systemContent += ` Focus on ${knowledgeFocus} when responding.`;
      }
      
      if (citeSources) {
        systemContent += " Cite sources when providing information.";
      }
      
      requestMessages.unshift({
        role: "system",
        content: systemContent
      });
    }
    
    return this.sendChatRequest({
      model: CHAT_SETTINGS.model,
      messages: requestMessages,
      max_tokens: CHAT_SETTINGS.max_tokens,
      temperature: CHAT_SETTINGS.temperature,
      top_p: CHAT_SETTINGS.top_p,
      frequency_penalty: CHAT_SETTINGS.frequency_penalty,
      presence_penalty: CHAT_SETTINGS.presence_penalty
    });
  }

  // New method to handle multi-agent API requests
  private async sendMultiAgentRequest(
    query: string,
    image?: string | null
  ): Promise<ChatCompletionResponse> {
    const requestId = this.generateRequestId();
    const startTime = Date.now();
    
    try {
      logger.info('Sending multi-agent request', { requestId, query });
      
      // Prepare the request payload
      const payload = {
        model: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages: [
          {
            role: "user",
            content: query
          }
        ],
        max_tokens: 4096,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0
      };
      
      // Send the initial request to start the task
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        throw new ApiError(
          `API request failed with status ${response.status}`,
          response.status,
          await response.json(),
          requestId
        );
      }
      
      const initialData = await response.json();
      
      // Extract the task ID from the response content
      const taskIdMatch = initialData.choices[0].message.content.match(/Task ID: ([a-zA-Z0-9-]+)/);
      if (!taskIdMatch) {
        throw new Error('Could not extract task ID from response');
      }
      
      const taskId = taskIdMatch[1];
      logger.info('Multi-agent task started', { requestId, taskId });
      
      // Poll for the result
      let result = null;
      let attempts = 0;
      const maxAttempts = 30; // Maximum number of polling attempts
      const pollInterval = 2000; // 2 seconds between polls
      
      while (attempts < maxAttempts) {
        await this.delay(pollInterval);
        attempts++;
        
        const statusResponse = await fetch(`${this.baseUrl}/v1/chat/status/${taskId}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (!statusResponse.ok) {
          logger.warn('Error checking task status', { 
            requestId, 
            taskId, 
            status: statusResponse.status 
          });
          continue;
        }
        
        const statusData = await statusResponse.json();
        
        if (statusData.status === 'completed') {
          result = statusData.response;
          break;
        } else if (statusData.status === 'error') {
          throw new Error(`Task failed: ${statusData.error}`);
        }
        
        logger.debug('Task still processing', { 
          requestId, 
          taskId, 
          attempt: attempts, 
          status: statusData.status 
        });
      }
      
      if (!result) {
        throw new Error('Task timed out after maximum polling attempts');
      }
      
      this.updateMetrics(startTime);
      return result;
      
    } catch (error) {
      this.updateMetrics(startTime, true);
      logger.error('Error in multi-agent request', { 
        requestId, 
        error: error instanceof Error ? error.message : String(error) 
      });
      throw error;
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