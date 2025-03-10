import { ApiClient } from '@/utils/apiClient';
import { clientConfig } from '@/config/clientConfig';
import { logger } from '@/utils/logger';

// Mock the clientConfig
jest.mock('@/config/clientConfig', () => ({
  clientConfig: {
    environment: 'test',
    version: '2.4.4',
    api: {
      baseUrl: '',
      endpoints: {
        chat: '/api/chat',
        ping: '/api/ping',
        health: '/api/health',
      },
      timeoutMs: 60000,
    },
    together: {
      endpoint: '/api/chat',
      model: 'test-model',
    },
    features: {
      enableAgenticMode: true,
      enableImageUploads: true,
      enableDevTools: false,
    },
    ui: {
      defaultTheme: 'dark',
      maxMessagesDisplayed: 50,
    },
  }
}));

// Mock the fetch API
global.fetch = jest.fn();

// Mock the logger
jest.mock('@/utils/logger', () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  }
}));

describe('ApiClient', () => {
  let apiClient: ApiClient;
  
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Initialize the API client
    apiClient = ApiClient.getInstance();
    apiClient.resetMetrics();
    
    // Setup default mock implementation for fetch
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: jest.fn().mockResolvedValue({
        choices: [
          {
            message: {
              content: 'Test response',
              role: 'assistant'
            },
            finish_reason: 'stop'
          }
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30
        }
      })
    });
  });
  
  test('should be a singleton', () => {
    const instance1 = ApiClient.getInstance();
    const instance2 = ApiClient.getInstance();
    
    expect(instance1).toBe(instance2);
  });
  
  test('should successfully make API requests', async () => {
    const response = await apiClient.sendChatRequest({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    });
    
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/chat'),
      expect.objectContaining({ 
        method: 'POST',
      })
    );
    
    expect(response).toEqual({
      choices: [
        {
          message: {
            content: 'Test response',
            role: 'assistant'
          },
          finish_reason: 'stop'
        }
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30
      }
    });
  });
  
  test('should handle API errors', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: jest.fn().mockResolvedValue({
        error: {
          message: 'Invalid request'
        }
      })
    });
    
    await expect(apiClient.sendChatRequest({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    })).rejects.toThrow('Invalid request');
    
    expect(logger.error).toHaveBeenCalled();
  });
  
  test('should retry on server errors', async () => {
    // First call fails with 500, second call succeeds
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Server Error',
        json: jest.fn().mockResolvedValue({
          error: {
            message: 'Internal server error'
          }
        })
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: jest.fn().mockResolvedValue({
          choices: [
            {
              message: {
                content: 'Retry succeeded',
                role: 'assistant'
              },
              finish_reason: 'stop'
            }
          ],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30
          }
        })
      });
    
    const response = await apiClient.sendChatRequest({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    });
    
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(response.choices[0].message.content).toBe('Retry succeeded');
    expect(logger.info).toHaveBeenCalledWith(
      expect.stringContaining('Retrying API request'),
      expect.any(Object)
    );
  });
  
  test('should not retry on client errors', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400, // Client error
      statusText: 'Bad Request',
      json: jest.fn().mockResolvedValue({
        error: {
          message: 'Invalid request parameters'
        }
      })
    });
    
    await expect(apiClient.sendChatRequest({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    })).rejects.toThrow('Invalid request parameters');
    
    expect(global.fetch).toHaveBeenCalledTimes(1); // No retry
  });
  
  test('should validate API responses', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: jest.fn().mockResolvedValue({
        // Invalid response missing required fields
        choices: []
      })
    });
    
    await expect(apiClient.sendChatRequest({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    })).rejects.toThrow('Invalid API response format');
    
    expect(logger.warn).toHaveBeenCalled();
  });
  
  test('should track metrics accurately', async () => {
    // Reset metrics to ensure a clean state
    apiClient.resetMetrics();
    
    // Make a successful request
    await apiClient.sendChatRequest({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    });
    
    // Make a failed request
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: jest.fn().mockResolvedValue({
        error: {
          message: 'Invalid request'
        }
      })
    });
    
    try {
      await apiClient.sendChatRequest({
        model: 'test-model',
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0
      });
    } catch (error) {
      // Expected error, ignore
    }
    
    const metrics = apiClient.getMetrics();
    
    expect(metrics.requestCount).toBe(2);
    expect(metrics.errorCount).toBe(1);
    expect(metrics.statusCodes['200']).toBe(1);
    expect(metrics.statusCodes['400']).toBe(1);
  });
  
  test('should handle network errors', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new TypeError('Network error'));
    
    await expect(apiClient.sendChatRequest({
      model: 'test-model',
      messages: [{ role: 'user', content: 'Hello' }],
      max_tokens: 100,
      temperature: 0.7,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    })).rejects.toThrow('Network error');
    
    expect(logger.error).toHaveBeenCalled();
  });
  
  test('should determine API health based on consecutive errors', () => {
    apiClient.resetMetrics();
    expect(apiClient.isHealthy()).toBe(true);
    
    // Simulate consecutive errors
    const metrics = apiClient.getMetrics();
    metrics.consecutiveErrors = 3;
    
    expect(apiClient.isHealthy()).toBe(false);
  });
}); 