'use client';

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { ChatMessage, ChatState, Source } from '@/types/chat';
import { storage } from '@/utils/storage';
import { CHAT_SETTINGS } from '@/config/chat';
import { apiClient } from '@/utils/apiClient';
import { rateLimiter } from '@/utils/rateLimiter';
import { logger } from '@/utils/logger';
import { useApp } from '@/contexts/AppContext';

export interface ChatContextType {
  state: ChatState;
  sendMessage: (message: string, imageFile?: File | null) => Promise<void>;
  clearMessages: () => void;
  getMessageHistory: () => ChatMessage[];
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// Function to generate mock sources based on message content
const generateMockSources = (content: string): Source[] => {
  // Only generate sources for certain types of content
  const shouldHaveSources = content.length > 100 && !content.includes("I don't know");
  
  if (!shouldHaveSources) {
    return [];
  }
  
  // Generate 3-5 mock sources (always have at least 3 for testing)
  const sourceCount = Math.floor(Math.random() * 3) + 3;
  
  const mockDomains = [
    'wikipedia.org', 
    'research.edu', 
    'science.gov', 
    'academic-journal.com',
    'nationalgeographic.com',
    'techreview.mit.edu',
    'nature.com',
    'arxiv.org',
    'medicalnews.org',
    'historyarchive.org'
  ];
  
  return Array.from({ length: sourceCount }, (_, i) => {
    // Extract some words from the content to create a realistic title
    const words = content.split(' ');
    const startIndex = Math.floor(Math.random() * (words.length - 10));
    const titleWords = words.slice(startIndex, startIndex + 8 + Math.floor(Math.random() * 5));
    const title = titleWords.join(' ').replace(/[.,;:!?]$/, '') + (Math.random() > 0.5 ? '' : ' - Research');
    
    // Extract a snippet from a different part of the content
    const snippetStart = Math.floor(Math.random() * Math.max(1, content.length - 200));
    const snippetLength = 100 + Math.floor(Math.random() * 100);
    const snippet = content.substring(snippetStart, snippetStart + snippetLength);
    
    // Pick a random domain
    const domain = mockDomains[Math.floor(Math.random() * mockDomains.length)];
    
    return {
      id: `source-${Date.now()}-${i}`,
      title,
      url: Math.random() > 0.2 ? `https://${domain}/article/${Date.now()}${i}` : undefined,
      snippet,
      relevance: 0.5 + (Math.random() * 0.5), // 0.5-1.0 relevance
      domain
    };
  });
};

// Helper function to convert file to base64
const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
      } else {
        reject(new Error('Failed to convert file to base64'));
      }
    };
    reader.onerror = error => reject(error);
  });
};

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const { accessibility } = useApp();
  const [state, setState] = useState<ChatState>(() => ({
    messages: [],
    isLoading: false,
  }));

  // Debug logging for accessibility settings
  useEffect(() => {
    console.log('Accessibility settings in ChatContext:', accessibility);
  }, [accessibility]);

  // Load messages from storage on client side
  useEffect(() => {
    try {
      setState(prev => ({
        ...prev,
        messages: storage.getMessages(),
      }));
      logger.info('Loaded messages from storage', { messageCount: state.messages.length });
    } catch (error) {
      logger.error('Failed to load messages from storage', error);
      // Fallback to empty messages array
      setState(prev => ({ ...prev, messages: [] }));
    }
  }, []);

  // Save messages to storage when they change
  useEffect(() => {
    try {
      storage.saveMessages(state.messages);
      logger.debug('Saved messages to storage', { messageCount: state.messages.length });
    } catch (error) {
      logger.error('Failed to save messages to storage', error);
    }
  }, [state.messages]);

  const handleError = useCallback((error: any, requestId: string) => {
    logger.error('Chat error occurred', { error, requestId });
    
    let errorMessage = 'An unexpected error occurred';
    if (error instanceof Error) {
      errorMessage = error.message;
    }

    setState(prev => ({
      ...prev,
      isLoading: false,
      error: errorMessage,
    }));
  }, []);

  const sendMessage = useCallback(
    async (message: string, imageFile?: File | null) => {
      if (state.isLoading) return;
      if (!message.trim() && !imageFile) return;

      let imageUrl: string | null = null;

      try {
        setState(prev => ({ ...prev, error: undefined, isLoading: true }));

        // Create a user message from the input
        const userMessage: ChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: message,
          timestamp: new Date().toISOString(),
        };

        // Update messages with user message
        const newMessages = [...state.messages, userMessage];
        setState(prev => ({ ...prev, messages: newMessages }));
        storage.saveMessages(newMessages);

        // Process image if provided
        if (imageFile) {
          // Create a URL for the image to display in the UI
          imageUrl = URL.createObjectURL(imageFile);

          // Convert image to base64 for the API
          const base64 = await fileToBase64(imageFile);
          imageUrl = base64;
        }

        // Check if rate limited
        if (rateLimiter.isRateLimited()) {
          const timeUntilNext = rateLimiter.getTimeUntilNextAllowed();
          throw new Error(`Rate limit exceeded. Please try again in ${Math.ceil(timeUntilNext / 1000)} seconds`);
        }

        // Track the request for rate limiting
        rateLimiter.addRequest();

        // Get response from API client
        const client = apiClient; 
        
        let response;
        try {
          logger.info('Sending message to API', { 
            agentic: accessibility.agentic, 
            promptStyle: accessibility.promptStyle,
            knowledgeFocus: accessibility.knowledgeFocus
          });
          
          response = await client.sendChatMessage(
            newMessages, 
            imageUrl, 
            accessibility.promptStyle, 
            accessibility.knowledgeFocus, 
            accessibility.citeSources,
            accessibility.agentic // Pass the agentic setting
          );
        } catch (error) {
          // Handle rate limit errors
          if (error instanceof Error && error.message.includes('Rate limit')) {
            setState(prev => ({ 
              ...prev, 
              error: error.message,
              isLoading: false
            }));
            return;
          }
          throw error;
        }

        // Check for error in the response (when using the multi-agent backend)
        if (response.choices[0].finish_reason === 'error') {
          throw new Error(response.choices[0].message.content);
        }
        
        let sources: Source[] = [];
        // If citeSources is enabled and we're not in agentic mode, generate mock sources
        if (accessibility.citeSources && !accessibility.agentic) {
          sources = generateMockSources(response.choices[0].message.content);
        }

        // Create assistant message
        const assistantMessage: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.choices[0].message.content,
          timestamp: new Date().toISOString(),
          sources: sources
        };
        
        logger.info('Received API response', { 
          responseLength: assistantMessage.content.length,
          sourceCount: assistantMessage.sources?.length || 0,
          agentic: accessibility.agentic
        });

        // Update state with the new message
        const updatedMessages = [...newMessages, assistantMessage];
        setState(prev => ({
          ...prev,
          messages: updatedMessages,
          isLoading: false,
        }));

        // Save to storage
        storage.saveMessages(updatedMessages);

        logger.info('Updated chat state with response', { 
          responseLength: assistantMessage.content.length,
          sourceCount: assistantMessage.sources?.length || 0,
          messageCount: updatedMessages.length
        });
      } catch (error) {
        logger.error('Error in sendMessage:', error);

        setState(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error 
            ? error.message 
            : 'An unknown error occurred'
        }));
      }
    },
    [state.messages, accessibility.promptStyle, accessibility.knowledgeFocus, accessibility.citeSources, accessibility.agentic]
  );

  const clearMessages = useCallback(() => {
    try {
      setState(prev => ({
        ...prev,
        messages: [],
        error: undefined,
      }));
      storage.saveMessages([]);
      logger.info('Cleared all messages');
    } catch (error) {
      logger.error('Failed to clear messages', error);
      handleError(error, 'clear_messages');
    }
  }, [handleError]);

  const getMessageHistory = useCallback((): ChatMessage[] => {
    return state.messages;
  }, [state.messages]);

  const value = {
    state,
    sendMessage,
    clearMessages,
    getMessageHistory,
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
} 