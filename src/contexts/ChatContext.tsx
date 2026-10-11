'use client';

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { ChatMessage, ChatState } from '@/types/chat';
import { ChatCompletionResponse } from '@/types/api';
import { storage } from '@/utils/storage';
import { apiClient } from '@/utils/apiClient';
import { logger } from '@/utils/logger';
import { useApp } from '@/contexts/AppContext';

export interface ChatContextType {
  state: ChatState;
  sendMessage: (message: string, imageFile?: File | null) => Promise<void>;
  clearMessages: () => void;
  getMessageHistory: () => ChatMessage[];
  researchInDepth: (assistantMessageId: string) => Promise<void>;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

let messageCounter = 0;

// Date.now() alone repeats within a millisecond; researchInDepth looks messages up by id.
function newMessageId(role: 'user' | 'assistant'): string {
  messageCounter += 1;
  return `${role}-${Date.now()}-${messageCounter}`;
}

// Sources and workflow logs belong only to answers from the research agents.
function buildAssistantMessage(response: ChatCompletionResponse): ChatMessage {
  const { message } = response.choices[0];
  const agentic = response.route?.route === 'agentic';
  const assistantMessage: ChatMessage = {
    id: newMessageId('assistant'),
    role: 'assistant',
    content: message.content,
    timestamp: new Date().toISOString(),
    route: response.route,
  };
  if (agentic && message.sources) assistantMessage.sources = message.sources;
  if (agentic && message.logs) assistantMessage.logs = message.logs;
  return assistantMessage;
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const { accessibility, agenticMode } = useApp();
  const [state, setState] = useState<ChatState>(() => ({
    messages: [],
    isLoading: false,
  }));
  const researchInFlightRef = useRef<string | null>(null);

  useEffect(() => {
    try {
      const loadedMessages = storage.getMessages();
      setState((prev) => ({
        ...prev,
        messages: loadedMessages,
      }));
      logger.info('Loaded messages from storage', { messageCount: loadedMessages.length });
    } catch (error) {
      logger.error('Failed to load messages from storage', error);
      setState((prev) => ({ ...prev, messages: [] }));
    }
  }, []);

  useEffect(() => {
    try {
      storage.saveMessages(state.messages);
      logger.debug('Saved messages to storage', { messageCount: state.messages.length });
    } catch (error) {
      logger.error('Failed to save messages to storage', error);
    }
  }, [state.messages, state.messages.length]);

  const handleError = useCallback((error: unknown, requestId: string) => {
    logger.error('Chat error occurred', { error, requestId });

    let errorMessage = 'An unexpected error occurred';
    if (error instanceof Error) {
      errorMessage = error.message;
    }

    setState((prev) => ({
      ...prev,
      isLoading: false,
      error: errorMessage,
    }));
  }, []);

  const sendMessage = useCallback(
    async (message: string, imageFile?: File | null) => {
      if (state.isLoading) return;
      if (!message.trim() && !imageFile) return;

      const userMessage: ChatMessage = {
        id: newMessageId('user'),
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
        imageAttached: !!imageFile,
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        isLoading: true,
        error: undefined,
      }));

      try {
        let imageBase64: string | null = null;
        if (imageFile) {
          imageBase64 = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = reject;
            reader.readAsDataURL(imageFile);
          });
        }

        const response = await apiClient.sendChatMessage(
          [...state.messages, userMessage],
          imageBase64,
          accessibility.promptStyle,
          accessibility.knowledgeFocus,
          accessibility.citeSources,
          agenticMode
        );

        if (response.choices[0].finish_reason === 'error') {
          throw new Error(response.choices[0].message.content);
        }

        const assistantMessage = buildAssistantMessage(response);

        setState((prev) => {
          const updatedMessages = [...prev.messages, assistantMessage];
          storage.saveMessages(updatedMessages);
          return {
            ...prev,
            messages: updatedMessages,
            isLoading: false,
          };
        });
      } catch (error) {
        logger.error('Error in sendMessage:', error);

        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'An unknown error occurred',
        }));
      }
    },
    [
      state.messages,
      state.isLoading,
      accessibility.promptStyle,
      accessibility.knowledgeFocus,
      accessibility.citeSources,
      agenticMode,
    ]
  );

  const researchInDepth = useCallback(
    async (assistantMessageId: string) => {
      if (state.isLoading || researchInFlightRef.current === assistantMessageId) return;

      const assistantIndex = state.messages.findIndex((m) => m.id === assistantMessageId);
      if (assistantIndex < 0) return;

      let userIndex = -1;
      for (let i = assistantIndex - 1; i >= 0; i--) {
        if (state.messages[i].role === 'user') {
          userIndex = i;
          break;
        }
      }
      if (userIndex < 0) return;

      researchInFlightRef.current = assistantMessageId;
      setState((prev) => ({ ...prev, isLoading: true, error: undefined }));

      try {
        const messagesToSend = state.messages.slice(0, userIndex + 1);
        const response = await apiClient.sendChatMessage(
          messagesToSend,
          null,
          accessibility.promptStyle,
          accessibility.knowledgeFocus,
          accessibility.citeSources,
          agenticMode,
          { forceAgentic: true }
        );

        const assistantMessage = buildAssistantMessage(response);

        // Insert below the direct answer (and any earlier research answers to the same
        // question), not at the end, so it stays with its question when the chat has moved on.
        setState((prev) => {
          const messages = [...prev.messages];
          let insertAt = messages.findIndex((m) => m.id === assistantMessageId);
          if (insertAt < 0) {
            messages.push(assistantMessage);
          } else {
            insertAt += 1;
            while (insertAt < messages.length && messages[insertAt].role === 'assistant') {
              insertAt += 1;
            }
            messages.splice(insertAt, 0, assistantMessage);
          }
          return { ...prev, messages, isLoading: false };
        });
      } catch (error) {
        logger.error('Error in researchInDepth:', error);
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'An unknown error occurred',
        }));
      } finally {
        researchInFlightRef.current = null;
      }
    },
    [
      state.messages,
      state.isLoading,
      accessibility.promptStyle,
      accessibility.knowledgeFocus,
      accessibility.citeSources,
      agenticMode,
    ]
  );

  const clearMessages = useCallback(() => {
    try {
      setState((prev) => ({
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
    researchInDepth,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
