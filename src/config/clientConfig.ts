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
  version: '2.9.10',
  isProduction: typeof window !== 'undefined' 
    ? process.env.NODE_ENV === 'production' 
    : false,
  isDevelopment: typeof window !== 'undefined' 
    ? process.env.NODE_ENV === 'development' 
    : true,
  api: {
    // Use the VERCEL_URL environment variable in production, or default to relative path
    baseUrl: typeof window !== 'undefined' && process.env.NEXT_PUBLIC_VERCEL_URL
      ? `https://${process.env.NEXT_PUBLIC_VERCEL_URL}/api`
      : '/api',
    endpoints: {
      chat: '/chat',  // Direct LLM endpoint (non-agentic mode)
      task: '/task',  // Multi-agent endpoint (agentic mode)
      ping: '/ping',
      health: '/health',
    },
    timeoutMs: 60000, // 60 seconds
  },
  together: {
    // Note: API key is NOT included here for security reasons
    endpoint: '/api/chat', // Use our local API endpoint instead of direct access
    model: 'meta-llama/Llama-3.3-70b-instruct-turbo-free',
    fallbackModel: 'meta-llama/Llama-3.3-8b-instruct', // Smaller model as backup if needed
  },
  features: {
    enableAgenticMode: true,
    enableNonAgenticFallback: true, // Ensure non-agentic mode works as a fallback
    enableMultiAgentFallback: true, // Allow multi-agent system to fall back to direct LLM
    enableImageUploads: true,
    enableDevTools: typeof window !== 'undefined' 
      ? process.env.NODE_ENV !== 'production'
      : false,
  },
  ui: {
    defaultTheme: 'dark',
    maxMessagesDisplayed: 50,
  },
  name: 'WebAgent Platform',
  slogan: 'Your AI Research Assistant',
  logoText: 'WebAgent',
  footerText: '© 2025 WebAgent Platform',
} as const;

/**
 * Type representing the client configuration structure
 */
export type ClientConfig = typeof clientConfig; 