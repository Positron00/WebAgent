import { ChatCompletionRequest, ChatCompletionResponse } from '@/types/api';
import { ChatMessage } from '@/types/chat';
import { CHAT_SETTINGS } from '@/config/chat';
import { logger } from '@/utils/logger';
import { clientConfig } from '@/config/clientConfig';

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

/**
 * Enhanced error class with additional context for better debugging
 */
class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any,
    public requestId?: string,
    public endpoint?: string,
    public timestamp: number = Date.now()
  ) {
    super(message);
    this.name = 'ApiError';
    
    // Capture stack trace for better debugging
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }
    
    // Log the error with context
    logger.error('API Error', {
      message,
      status,
      requestId,
      endpoint,
      timestamp: new Date(timestamp).toISOString(),
      data: typeof data === 'object' ? JSON.stringify(data) : data
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
    statusCodes: {}
  };
  
  private constructor(options: Partial<RetryOptions> = {}) {
    this.retryOptions = {
      maxRetries: options.maxRetries ?? 3,
      initialDelay: options.initialDelay ?? 500,
      maxDelay: options.maxDelay ?? 5000,
      backoffFactor: options.backoffFactor ?? 2
    };
    
    logger.info('ApiClient initialized', { 
      retryOptions: this.retryOptions,
      baseUrl: this.baseUrl,
      environment: clientConfig.environment
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
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  
  private getRetryDelay(attempt: number): number {
    const delay = Math.min(
      this.retryOptions.initialDelay * Math.pow(this.retryOptions.backoffFactor, attempt),
      this.retryOptions.maxDelay
    );
    // Add jitter to prevent synchronized retries
    return delay * (0.8 + Math.random() * 0.4);
  }
  
  private isRetryableError(error: any): boolean {
    // Network errors should be retried
    if (error instanceof TypeError && error.message.includes('network')) {
      return true;
    }
    
    // Retry server errors (5xx) but not client errors (4xx)
    if (error instanceof ApiError) {
      return error.status >= 500 && error.status < 600;
    }
    
    // Timeout errors should be retried
    if (error instanceof Error && error.message.includes('timeout')) {
      return true;
    }
    
    return false;
  }
  
  private validateResponse(data: any): data is ChatCompletionResponse {
    const errors = this.getValidationErrors(data);
    
    if (errors.length > 0) {
      logger.warn('Invalid API response', { errors, data: JSON.stringify(data).substring(0, 500) });
      return false;
    }
    
    return true;
  }
  
  private updateMetrics(startTime: number, isError: boolean = false, retries: number = 0, statusCode?: number) {
    const requestTime = Date.now() - startTime;
    
    this.metrics.requestCount++;
    this.metrics.totalResponseTime += requestTime;
    this.metrics.averageResponseTime = this.metrics.totalResponseTime / this.metrics.requestCount;
    this.metrics.lastRequestTime = Date.now();
    
    if (statusCode) {
      const statusCodeKey = `${statusCode}`;
      this.metrics.statusCodes[statusCodeKey] = (this.metrics.statusCodes[statusCodeKey] || 0) + 1;
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
      logger.info('API Metrics', {
        ...this.metrics,
        errorRate: this.metrics.errorCount / this.metrics.requestCount,
        retryRate: this.metrics.retryCount / this.metrics.requestCount
      });
    }
  }
  
  async fetchWithRetry<T>(
    url: string,
    options: RequestInit,
    customRetryOptions?: Partial<RetryOptions>
  ): Promise<T> {
    const retryOptions: RetryOptions = {
      ...this.retryOptions,
      ...customRetryOptions
    };
    
    const requestId = this.generateRequestId();
    const startTime = Date.now();
    const fullUrl = url.startsWith('http') ? url : `${this.baseUrl}${url}`;
    let attempt = 0;
    
    // Add request ID to headers for tracing
    const headers = new Headers(options.headers || {});
    headers.set('X-Request-ID', requestId);
    
    // Set timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, clientConfig.api.timeoutMs);
    
    logger.debug('API request', { 
      method: options.method, 
      url: fullUrl, 
      requestId,
      attempt: 0
    });
    
    while (true) {
      try {
        const response = await fetch(fullUrl, {
          ...options,
          headers,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Update metrics with status code
        this.updateMetrics(startTime, !response.ok, attempt, response.status);
        
        if (!response.ok) {
          let errorData;
          try {
            errorData = await response.json();
          } catch (e) {
            errorData = { error: response.statusText };
          }
          
          throw new ApiError(
            errorData.error?.message || `HTTP Error: ${response.statusText}`,
            response.status,
            errorData,
            requestId,
            fullUrl
          );
        }
        
        const data = await response.json();
        
        // For chat completions, validate the response structure
        if (url.includes('/chat') && !this.validateResponse(data)) {
          throw new ApiError(
            'Invalid API response format',
            200,
            data,
            requestId,
            fullUrl
          );
        }
        
        logger.debug('API response', { 
          method: options.method, 
          url: fullUrl, 
          requestId,
          statusCode: response.status,
          duration: Date.now() - startTime,
          attempt
        });
        
        return data;
      } catch (error: any) {
        const isRetryable = this.isRetryableError(error);
        const canRetry = attempt < retryOptions.maxRetries && isRetryable;
        
        clearTimeout(timeoutId);
        
        // Log error details
        logger.error('API request failed', {
          method: options.method,
          url: fullUrl,
          requestId,
          attempt,
          error: error.message,
          status: error instanceof ApiError ? error.status : undefined,
          isRetryable,
          canRetry,
          retryCount: attempt,
          maxRetries: retryOptions.maxRetries
        });
        
        if (!canRetry) {
          this.updateMetrics(startTime, true, attempt);
          throw error;
        }
        
        attempt++;
        const delayMs = this.getRetryDelay(attempt);
        
        logger.info('Retrying API request', {
          method: options.method,
          url: fullUrl,
          requestId,
          attempt,
          delayMs
        });
        
        await this.delay(delayMs);
      }
    }
  }
  
  private getValidationErrors(data: any): string[] {
    const errors = [];
    
    if (typeof data !== 'object' || data === null) errors.push('response is not an object');
    if (!Array.isArray(data?.choices)) errors.push('choices is not an array');
    if (data?.choices?.length === 0) errors.push('choices array is empty');
    if (typeof data?.choices?.[0] !== 'object') errors.push('first choice is not an object');
    if (typeof data?.choices?.[0]?.message !== 'object') errors.push('message is not an object');
    if (typeof data?.choices?.[0]?.message?.content !== 'string') errors.push('message content is not a string');
    if (typeof data?.choices?.[0]?.message?.role !== 'string') errors.push('message role is not a string');
    return errors;
  }

  async sendChatRequest(
    request: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    return this.fetchWithRetry<ChatCompletionResponse>(
      clientConfig.api.endpoints.chat,
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
    
    // Legacy non-agentic path (fallback) - use our API route instead of direct access
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
    
    // Send to our Next.js API route instead of directly to Together AI
    return this.fetchWithRetry<ChatCompletionResponse>(
      clientConfig.api.endpoints.chat,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: requestMessages,
          model: clientConfig.together.model,
          max_tokens: CHAT_SETTINGS.maxTokens,
          temperature: CHAT_SETTINGS.temperature,
          top_p: CHAT_SETTINGS.topP,
          frequency_penalty: CHAT_SETTINGS.frequencyPenalty,
          presence_penalty: CHAT_SETTINGS.presencePenalty,
          // Include the settings needed by the API route
          promptStyle,
          knowledgeFocus,
          citeSources,
          agentic,
          // Include image if provided
          image: image || null
        }),
      }
    );
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
      const requestBody = {
        query,
        image: image || undefined
      };
      
      // Use the multi-agent API endpoint
      const response = await this.fetchWithRetry<ChatCompletionResponse>(
        '/api/task', // Different endpoint for multi-agent requests
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Request-ID': requestId
          },
          body: JSON.stringify(requestBody)
        }
      );
      
      // Check if we need to wait for an async task to complete
      if (response.choices?.[0]?.message?.content?.includes('task_')) {
        // This is a task ID, we need to poll for results
        return this.pollTaskResult(response.choices[0].message.content);
      }
      
      return response;
    } catch (error) {
      logger.error('Multi-agent request failed', { 
        requestId, 
        error: error instanceof Error ? error.message : String(error),
        query: query.substring(0, 100) // Only log the beginning of the query for privacy
      });
      
      // Create a fallback response for better user experience
      const fallbackResponse: ChatCompletionResponse = {
        choices: [{
          message: {
            role: 'assistant',
            content: 'I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists.'
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      // Only throw in development, provide fallback in production
      if (clientConfig.isDevelopment) {
        throw error;
      } else {
        return fallbackResponse;
      }
    }
  }
  
  /**
   * Poll for task results when using the async multi-agent system
   */
  private async pollTaskResult(taskId: string): Promise<ChatCompletionResponse> {
    const maxPolls = 30; // Maximum number of polling attempts
    const pollInterval = 1000; // Start with 1 second intervals
    const maxPollInterval = 5000; // Maximum 5 second intervals
    const pollBackoffFactor = 1.5; // Increase interval by 50% each time
    
    let pollCount = 0;
    let currentInterval = pollInterval;
    
    while (pollCount < maxPolls) {
      await this.delay(currentInterval);
      pollCount++;
      
      try {
        const result = await this.fetchWithRetry<any>(
          `/api/task/${taskId}`,
          { method: 'GET' }
        );
        
        // Check if the task is complete
        if (result.status === 'completed' && result.result) {
          // Convert task result to ChatCompletionResponse format
          return {
            choices: [{
              message: {
                role: 'assistant',
                content: result.result.content || result.result
              },
              finish_reason: 'stop'
            }],
            usage: result.usage || {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0
            }
          };
        }
        
        // If failed, return an error message
        if (result.status === 'failed') {
          throw new Error(`Task failed: ${result.error || 'Unknown error'}`);
        }
        
        // Increase polling interval with backoff
        currentInterval = Math.min(
          currentInterval * pollBackoffFactor,
          maxPollInterval
        );
        
        logger.info('Polling task', { 
          taskId, 
          pollCount, 
          status: result.status,
          nextPollIn: currentInterval
        });
      } catch (error) {
        logger.error('Error polling task', { 
          taskId, 
          pollCount, 
          error: error instanceof Error ? error.message : String(error) 
        });
        
        // Increase polling interval on error
        currentInterval = Math.min(
          currentInterval * 2,
          maxPollInterval
        );
      }
    }
    
    // If we've reached the maximum polling attempts, return a timeout error
    throw new Error(`Task timed out after ${maxPolls} polling attempts`);
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
      statusCodes: {}
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
    if (this.metrics.requestCount > 5 && 
        this.metrics.errorCount / this.metrics.requestCount > 0.5) {
      return false;
    }
    
    return true;
  }
}

export const apiClient = ApiClient.getInstance(); 