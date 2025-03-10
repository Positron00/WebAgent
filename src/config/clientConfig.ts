// Client-safe configuration
// This file DOES NOT include any sensitive information like API keys

/**
 * Client-safe configuration that can be safely imported in browser-rendered components.
 * This configuration NEVER includes sensitive data like API keys.
 */
export const clientConfig = {
  environment: typeof window !== 'undefined' 
    ? (process.env.NODE_ENV || 'development') 
    : 'server',
  version: '2.4.4',
  isProduction: typeof window !== 'undefined' 
    ? process.env.NODE_ENV === 'production' 
    : false,
  isDevelopment: typeof window !== 'undefined' 
    ? process.env.NODE_ENV === 'development' 
    : true,
  api: {
    baseUrl: '/api',
    endpoints: {
      chat: '/chat',
      ping: '/ping',
      health: '/health',
    },
    timeoutMs: 60000, // 60 seconds
  },
  together: {
    // Note: API key is NOT included here for security reasons
    endpoint: '/api/chat', // Use our local API endpoint instead of direct access
    model: 'meta-llama/Llama-3.3-70b-instruct-turbo-free',
  },
  features: {
    enableAgenticMode: true,
    enableImageUploads: true,
    enableDevTools: typeof window !== 'undefined' 
      ? process.env.NODE_ENV !== 'production'
      : false,
  },
  ui: {
    defaultTheme: 'dark',
    maxMessagesDisplayed: 50,
  },
} as const;

/**
 * Type representing the client configuration structure
 */
export type ClientConfig = typeof clientConfig; 