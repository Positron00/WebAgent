"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useRef,
} from "react";
import { ChatMessage, ChatState } from "@/types/chat";
import { ChatCompletionResponse } from "@/types/api";
import { RESEARCH_STOPPED_MESSAGE, ResearchFailed } from "@/types/research";
import { storage, STORAGE_QUOTA_NOTICE } from "@/utils/storage";
import { apiClient, ApiError } from "@/utils/apiClient";
import { stoppedError } from "@/utils/taskPolling";
import { logger } from "@/utils/logger";
import { useApp } from "@/contexts/AppContext";
import {
  ResearchFailure,
  ResearchRun,
  RetryInfo,
  useResearchRun,
} from "@/hooks/useResearchRun";
import {
  GuardRefusal,
  isGuardRefusal,
  normalizeGuardRefusal,
} from "@/types/security";

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
  researchRun: ResearchRun | null;
  researchFailure: ResearchFailure | null;
  sendMessage: (message: string, imageFile?: File | null) => Promise<void>;
  sendWithConsent: () => Promise<void>;
  editPendingMessage: () => void;
  setEditPendingHandler: (handler: (text: string) => void) => void;
  clearMessages: () => void;
  getMessageHistory: () => ChatMessage[];
  researchInDepth: (assistantMessageId: string) => Promise<void>;
  approveAndRerun: (assistantMessageId: string) => Promise<void>;
  stopResearch: () => Promise<void>;
  retryResearch: () => Promise<void>;
  answerDirectlyInstead: () => Promise<void>;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

let messageCounter = 0;

function newMessageId(role: "user" | "assistant"): string {
  messageCounter += 1;
  return `${role}-${Date.now()}-${messageCounter}`;
}

function buildAssistantMessage(response: ChatCompletionResponse): ChatMessage {
  const { message } = response.choices[0];
  const agentic = response.route?.route === "agentic";
  const assistantMessage: ChatMessage = {
    id: newMessageId("assistant"),
    role: "assistant",
    content: message.content,
    timestamp: new Date().toISOString(),
    route: response.route,
  };
  if (agentic && message.sources) assistantMessage.sources = message.sources;
  if (agentic && message.logs) assistantMessage.logs = message.logs;
  if (response.security) assistantMessage.security = response.security;
  return assistantMessage;
}

function noteMessage(content: string, refusal?: GuardRefusal): ChatMessage {
  return {
    id: newMessageId("assistant"),
    role: "assistant",
    content,
    timestamp: new Date().toISOString(),
    ...(refusal ? { refusal } : {}),
  };
}

function guardRefusalFromError(error: unknown): GuardRefusal | null {
  if (!(error instanceof ApiError) || !isGuardRefusal(error.data)) {
    return null;
  }
  const raw = error.data as unknown as Record<string, unknown>;
  return (
    normalizeGuardRefusal(raw, error.status) ?? {
      code: error.data.code,
      reason: typeof raw.reason === "string" ? raw.reason : error.message,
      consentToken:
        typeof raw.consent_token === "string"
          ? raw.consent_token
          : typeof raw.consentToken === "string"
            ? raw.consentToken
            : undefined,
    }
  );
}

function readImage(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/** Index of the question an assistant message answers, or -1. */
function questionIndexFor(
  messages: ChatMessage[],
  assistantMessageId: string,
): number {
  const assistantIndex = messages.findIndex((m) => m.id === assistantMessageId);
  for (let i = assistantIndex - 1; i >= 0; i--) {
    if (messages[i].role === "user") return i;
  }
  return -1;
}

const errorText = (error: unknown) =>
  error instanceof Error ? error.message : "An unknown error occurred";

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const { accessibility, agenticMode, showNotification } = useApp();
  const [state, setState] = useState<ChatState>(() => ({
    messages: [],
    isLoading: false,
  }));
  const [pendingConsent, setPendingConsent] = useState<PendingConsent | null>(
    null,
  );
  const researchInFlightRef = useRef<string | null>(null);
  const editPendingHandlerRef = useRef<((text: string) => void) | null>(null);
  const {
    run: researchRun,
    failure: researchFailure,
    begin: beginResearch,
    stop: stopResearchRun,
    fail: failResearch,
    clear: clearResearch,
  } = useResearchRun();

  useEffect(() => {
    try {
      const loadedMessages = storage.getMessages();
      setState((prev) => ({
        ...prev,
        messages: loadedMessages,
      }));
      logger.info("Loaded messages from storage", {
        messageCount: loadedMessages.length,
      });
    } catch (error) {
      logger.error("Failed to load messages from storage", error);
      setState((prev) => ({ ...prev, messages: [] }));
    }
  }, []);

  useEffect(() => {
    const { persisted, quotaTrimmed } = storage.saveMessages(state.messages);
    if (quotaTrimmed) {
      showNotification({
        message: STORAGE_QUOTA_NOTICE,
        type: "warning",
      });
      if (persisted.length !== state.messages.length) {
        setState((prev) => ({ ...prev, messages: persisted }));
      }
    } else {
      logger.debug("Saved messages to storage", {
        messageCount: state.messages.length,
      });
    }
  }, [state.messages, state.messages.length, showNotification]);

  const setEditPendingHandler = useCallback(
    (handler: (text: string) => void) => {
      editPendingHandlerRef.current = handler;
    },
    [],
  );

  /** Adds an assistant message at the end, or under `anchorId` (after any answers already there). */
  const placeAssistant = useCallback(
    (assistantMessage: ChatMessage, anchorId?: string) => {
      setState((prev) => {
        const messages = [...prev.messages];
        let insertAt = anchorId
          ? messages.findIndex((m) => m.id === anchorId)
          : -1;
        if (insertAt < 0) {
          messages.push(assistantMessage);
        } else {
          insertAt += 1;
          while (
            insertAt < messages.length &&
            messages[insertAt].role === "assistant"
          ) {
            insertAt += 1;
          }
          messages.splice(insertAt, 0, assistantMessage);
        }
        return { ...prev, messages, isLoading: false, error: undefined };
      });
    },
    [],
  );

  /** Shows a guard refusal (consent dialog or placeholder) or the error banner. */
  const handleRequestError = useCallback(
    (error: unknown, request: RetryInfo) => {
      const refusal = guardRefusalFromError(error);
      const userMessage = request.messages[request.messages.length - 1];
      if (
        refusal?.code === "consent_required" &&
        refusal.consentToken &&
        !request.anchorId
      ) {
        setPendingConsent({
          messageId: userMessage.id!,
          text: userMessage.content,
          imageBase64: request.imageBase64,
          consentToken: refusal.consentToken,
          reason: refusal.reason,
        });
        setState((prev) => ({ ...prev, isLoading: false, error: undefined }));
        return;
      }
      if (refusal) {
        placeAssistant(noteMessage("", refusal), request.anchorId);
        return;
      }
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorText(error),
      }));
    },
    [placeAssistant],
  );

  /** Sends one request. Research failures are handled here; other errors are thrown. */
  const runChatRequest = useCallback(
    async (request: RetryInfo) => {
      const mode = request.mode ?? agenticMode;
      const research = beginResearch();
      try {
        const response = await apiClient.sendChatMessage(
          request.messages,
          request.imageBase64,
          accessibility.promptStyle,
          accessibility.knowledgeFocus,
          accessibility.citeSources,
          mode,
          { ...request.options, research },
        );
        if (research.signal.aborted) throw stoppedError();
        if (response.choices[0].finish_reason === "error") {
          throw new Error(response.choices[0].message.content);
        }
        clearResearch();
        placeAssistant(buildAssistantMessage(response), request.anchorId);
      } catch (error) {
        const failed = research.signal.aborted ? stoppedError() : error;
        if (!(failed instanceof ResearchFailed)) {
          clearResearch();
          throw failed;
        }
        if (failed.kind === "cancelled") {
          clearResearch();
          placeAssistant(
            noteMessage(RESEARCH_STOPPED_MESSAGE),
            request.anchorId,
          );
          return;
        }
        failResearch(failed.kind, failed.message, { ...request, mode });
        setState((prev) => ({ ...prev, isLoading: false }));
      }
    },
    [
      accessibility.promptStyle,
      accessibility.knowledgeFocus,
      accessibility.citeSources,
      agenticMode,
      beginResearch,
      clearResearch,
      failResearch,
      placeAssistant,
    ],
  );

  /** Shows the loading state, runs the request and reports any error. */
  const run = useCallback(
    async (request: RetryInfo, where: string) => {
      setState((prev) => ({ ...prev, isLoading: true, error: undefined }));
      try {
        await runChatRequest(request);
      } catch (error) {
        logger.error(`Error in ${where}:`, error);
        handleRequestError(error, request);
      }
    },
    [runChatRequest, handleRequestError],
  );

  const sendMessage = useCallback(
    async (message: string, imageFile?: File | null) => {
      if (state.isLoading) return;
      if (!message.trim() && !imageFile) return;

      const userMessage: ChatMessage = {
        id: newMessageId("user"),
        role: "user",
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

      const messages = [...state.messages, userMessage];
      let imageBase64: string | null = null;
      try {
        if (imageFile) imageBase64 = await readImage(imageFile);
      } catch (error) {
        logger.error("Could not read the image:", error);
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: errorText(error),
        }));
        return;
      }
      await run({ messages, imageBase64 }, "sendMessage");
    },
    [state.messages, state.isLoading, run],
  );

  const sendWithConsent = useCallback(async () => {
    if (!pendingConsent || state.isLoading) return;

    const pending = pendingConsent;
    const userIndex = state.messages.findIndex(
      (m) => m.id === pending.messageId,
    );
    setPendingConsent(null);
    if (userIndex < 0) {
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: "Pending message not found",
      }));
      return;
    }
    await run(
      {
        messages: state.messages.slice(0, userIndex + 1),
        imageBase64: pending.imageBase64,
        options: { consentToken: pending.consentToken },
      },
      "sendWithConsent",
    );
  }, [pendingConsent, state.isLoading, state.messages, run]);

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

  /** Asks the same question again with research forced; the answer goes under the old one. */
  const rerunBelow = useCallback(
    async (
      assistantMessageId: string,
      options: RetryInfo["options"],
      where: string,
    ) => {
      if (state.isLoading || researchInFlightRef.current === assistantMessageId)
        return;
      const userIndex = questionIndexFor(state.messages, assistantMessageId);
      if (userIndex < 0) return;

      researchInFlightRef.current = assistantMessageId;
      try {
        await run(
          {
            messages: state.messages.slice(0, userIndex + 1),
            imageBase64: null,
            options,
            anchorId: assistantMessageId,
          },
          where,
        );
      } finally {
        researchInFlightRef.current = null;
      }
    },
    [state.messages, state.isLoading, run],
  );

  const researchInDepth = useCallback(
    (assistantMessageId: string) =>
      rerunBelow(assistantMessageId, { forceAgentic: true }, "researchInDepth"),
    [rerunBelow],
  );

  const approveAndRerun = useCallback(
    async (assistantMessageId: string) => {
      const assistantMessage = state.messages.find(
        (m) => m.id === assistantMessageId,
      );
      const approvals = assistantMessage?.security?.approvals ?? [];
      if (approvals.length === 0) return;
      const approvalTokens = approvals
        .map((a) => a.approvalToken)
        .filter((token) => token.length > 0);
      await rerunBelow(
        assistantMessageId,
        { forceAgentic: true, approvalTokens },
        "approveAndRerun",
      );
    },
    [state.messages, rerunBelow],
  );

  const stopResearch = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: false }));
    await stopResearchRun();
  }, [stopResearchRun]);

  const rerunFailed = useCallback(
    async (direct: boolean) => {
      const retry = researchFailure?.retry;
      if (!retry || state.isLoading) return;
      const request: RetryInfo = direct
        ? {
            ...retry,
            mode: "off",
            options: { consentToken: retry.options?.consentToken },
          }
        : retry;
      await run(request, direct ? "answerDirectlyInstead" : "retryResearch");
    },
    [researchFailure, state.isLoading, run],
  );

  const retryResearch = useCallback(() => rerunFailed(false), [rerunFailed]);
  const answerDirectlyInstead = useCallback(
    () => rerunFailed(true),
    [rerunFailed],
  );

  const clearMessages = useCallback(() => {
    try {
      setState((prev) => ({
        ...prev,
        messages: [],
        error: undefined,
      }));
      setPendingConsent(null);
      if (!researchRun) clearResearch();
      storage.saveMessages([]);
      logger.info("Cleared all messages");
    } catch (error) {
      logger.error("Failed to clear messages", error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error:
          error instanceof Error ? error.message : "Failed to clear messages",
      }));
    }
  }, [researchRun, clearResearch]);

  const getMessageHistory = useCallback((): ChatMessage[] => {
    return state.messages;
  }, [state.messages]);

  const value = {
    state,
    pendingConsent,
    researchRun,
    researchFailure,
    sendMessage,
    sendWithConsent,
    editPendingMessage,
    setEditPendingHandler,
    clearMessages,
    getMessageHistory,
    researchInDepth,
    approveAndRerun,
    stopResearch,
    retryResearch,
    answerDirectlyInstead,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
}
