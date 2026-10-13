'use client';

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import { ChatMessage, ChatState } from '@/types/chat';
import { ChatCompletionResponse } from '@/types/api';
import { storage } from '@/utils/storage';
import { apiClient, ApiError } from '@/utils/apiClient';
import { logger } from '@/utils/logger';
import { useApp } from '@/contexts/AppContext';
import { GuardRefusal, isGuardRefusal, normalizeGuardRefusal } from '@/types/security';

export interface PendingConsent {
  messageId: string;
  text: string;
  imageBase64: string | null;
  consentToken: string;
  reason: string;
}

export interface ChatContextType {
  state: ChatState;
  pendingConsent: PendingConsent | null;
  sendMessage: (message: string, imageFile?: File | null) => Promise<void>;
  sendWithConsent: () => Promise<void>;
  editPendingMessage: () => void;
  setEditPendingHandler: (handler: (text: string) => void) => void;
  clearMessages: () => void;
  getMessageHistory: () => ChatMessage[];
  researchInDepth: (assistantMessageId: string) => Promise<void>;
  approveAndRerun: (assistantMessageId: string) => Promise<void>;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

let messageCounter = 0;

function newMessageId(role: 'user' | 'assistant'): string {
  messageCounter += 1;
  return `${role}-${Date.now()}-${messageCounter}`;
}

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
  if (response.security) assistantMessage.security = response.security;
  return assistantMessage;
}

function refusalMessage(refusal: GuardRefusal): ChatMessage {
  return {
    id: newMessageId('assistant'),
    role: 'assistant',
    content: '',
    timestamp: new Date().toISOString(),
    refusal,
  };
}

function guardRefusalFromError(error: ApiError): GuardRefusal | null {
  if (!isGuardRefusal(error.data)) {
    return null;
  }
  const raw = error.data as unknown as Record<string, unknown>;
  return (
    normalizeGuardRefusal(raw, error.status) ?? {
      code: error.data.code,
      reason: typeof raw.reason === 'string' ? raw.reason : error.message,
      consentToken:
        typeof raw.consent_token === 'string'
          ? raw.consent_token
          : typeof raw.consentToken === 'string'
            ? raw.consentToken
            : undefined,
    }
  );
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const { accessibility, agenticMode } = useApp();
  const [state, setState] = useState<ChatState>(() => ({
    messages: [],
    isLoading: false,
  }));
  const [pendingConsent, setPendingConsent] = useState<PendingConsent | null>(null);
  const researchInFlightRef = useRef<string | null>(null);
  const editPendingHandlerRef = useRef<((text: string) => void) | null>(null);

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

  const setEditPendingHandler = useCallback((handler: (text: string) => void) => {
    editPendingHandlerRef.current = handler;
  }, []);

  const handleGuardRefusal = useCallback(
    (refusal: GuardRefusal, userMessage: ChatMessage, imageBase64: string | null) => {
      if (refusal.code === 'consent_required' && refusal.consentToken) {
        setPendingConsent({
          messageId: userMessage.id!,
          text: userMessage.content,
          imageBase64,
          consentToken: refusal.consentToken,
          reason: refusal.reason,
        });
        setState((prev) => ({ ...prev, isLoading: false, error: undefined }));
        return;
      }

      const placeholder = refusalMessage(refusal);
      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, placeholder],
        isLoading: false,
        error: undefined,
      }));
    },
    []
  );

  const runChatRequest = useCallback(
    async (
      messages: ChatMessage[],
      imageBase64: string | null,
      options?: { consentToken?: string; approvalTokens?: string[]; forceAgentic?: boolean }
    ) => {
      const response = await apiClient.sendChatMessage(
        messages,
        imageBase64,
        accessibility.promptStyle,
        accessibility.knowledgeFocus,
        accessibility.citeSources,
        agenticMode,
        options
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
    },
    [
      accessibility.promptStyle,
      accessibility.knowledgeFocus,
      accessibility.citeSources,
      agenticMode,
    ]
  );

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

        await runChatRequest([...state.messages, userMessage], imageBase64);
      } catch (error) {
        logger.error('Error in sendMessage:', error);

        if (error instanceof ApiError) {
          const refusal = guardRefusalFromError(error);
          if (refusal) {
            let imageBase64: string | null = null;
            if (imageFile) {
              imageBase64 = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result as string);
                reader.onerror = reject;
                reader.readAsDataURL(imageFile);
              });
            }
            handleGuardRefusal(refusal, userMessage, imageBase64);
            return;
          }
        }

        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'An unknown error occurred',
        }));
      }
    },
    [state.messages, state.isLoading, runChatRequest, handleGuardRefusal]
  );

  const sendWithConsent = useCallback(async () => {
    if (!pendingConsent || state.isLoading) return;

    const pending = pendingConsent;
    const userIndex = state.messages.findIndex((m) => m.id === pending.messageId);
    const userMessage = userIndex >= 0 ? state.messages[userIndex] : undefined;
    setPendingConsent(null);
    setState((prev) => ({ ...prev, isLoading: true, error: undefined }));

    try {
      if (!userMessage || userIndex < 0) {
        throw new Error('Pending message not found');
      }
      const historyBefore = state.messages.slice(0, userIndex + 1);

      await runChatRequest(historyBefore, pending.imageBase64, {
        consentToken: pending.consentToken,
      });
    } catch (error) {
      logger.error('Error in sendWithConsent:', error);
      if (error instanceof ApiError && userMessage) {
        const refusal = guardRefusalFromError(error);
        if (refusal) {
          handleGuardRefusal(refusal, userMessage, pending.imageBase64);
          return;
        }
      }
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'An unknown error occurred',
      }));
    }
  }, [pendingConsent, state.isLoading, state.messages, runChatRequest, handleGuardRefusal]);

  const editPendingMessage = useCallback(() => {
    if (!pendingConsent) return;
    const { messageId, text } = pendingConsent;
    setState((prev) => ({
      ...prev,
      messages: prev.messages.filter((m) => m.id !== messageId),
    }));
    editPendingHandlerRef.current?.(text);
    setPendingConsent(null);
  }, [pendingConsent]);

  const insertAssistantBelow = useCallback(
    (anchorAssistantId: string, assistantMessage: ChatMessage) => {
      setState((prev) => {
        const messages = [...prev.messages];
        let insertAt = messages.findIndex((m) => m.id === anchorAssistantId);
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
    },
    []
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
        insertAssistantBelow(assistantMessageId, assistantMessage);
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
      insertAssistantBelow,
    ]
  );

  const approveAndRerun = useCallback(
    async (assistantMessageId: string) => {
      if (state.isLoading || researchInFlightRef.current === assistantMessageId) return;

      const assistantMessage = state.messages.find((m) => m.id === assistantMessageId);
      const approvals = assistantMessage?.security?.approvals ?? [];
      if (approvals.length === 0) return;

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
        const approvalTokens = approvals
          .map((a) => a.approvalToken)
          .filter((token) => token.length > 0);

        const response = await apiClient.sendChatMessage(
          messagesToSend,
          null,
          accessibility.promptStyle,
          accessibility.knowledgeFocus,
          accessibility.citeSources,
          agenticMode,
          { forceAgentic: true, approvalTokens }
        );

        const newAssistant = buildAssistantMessage(response);
        insertAssistantBelow(assistantMessageId, newAssistant);
      } catch (error) {
        logger.error('Error in approveAndRerun:', error);
        if (error instanceof ApiError) {
          const refusal = guardRefusalFromError(error);
          if (refusal) {
            const placeholder = refusalMessage(refusal);
            insertAssistantBelow(assistantMessageId, placeholder);
            return;
          }
        }
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
      insertAssistantBelow,
    ]
  );

  const clearMessages = useCallback(() => {
    try {
      setState((prev) => ({
        ...prev,
        messages: [],
        error: undefined,
      }));
      setPendingConsent(null);
      storage.saveMessages([]);
      logger.info('Cleared all messages');
    } catch (error) {
      logger.error('Failed to clear messages', error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to clear messages',
      }));
    }
  }, []);

  const getMessageHistory = useCallback((): ChatMessage[] => {
    return state.messages;
  }, [state.messages]);

  const value = {
    state,
    pendingConsent,
    sendMessage,
    sendWithConsent,
    editPendingMessage,
    setEditPendingHandler,
    clearMessages,
    getMessageHistory,
    researchInDepth,
    approveAndRerun,
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
