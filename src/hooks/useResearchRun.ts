"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage } from "@/types/chat";
import { AgenticMode } from "@/types/routing";
import {
  ResearchFailureKind,
  ResearchProgress,
  ResearchRunOptions,
} from "@/types/research";
import { apiClient } from "@/utils/apiClient";
import { logger } from "@/utils/logger";

/** What is needed to send the same request again. */
export interface RetryInfo {
  messages: ChatMessage[];
  imageBase64: string | null;
  options?: {
    forceAgentic?: boolean;
    consentToken?: string;
    approvalTokens?: string[];
  };
  /** Put the answer under this assistant message instead of at the end. */
  anchorId?: string;
  /** The research mode the request used; the current mode when missing. */
  mode?: AgenticMode;
}

export interface ResearchFailure {
  kind: ResearchFailureKind;
  message: string;
  retry: RetryInfo;
}

/** A research task that exists on the backend and can be stopped. */
export interface ResearchRun {
  taskId: string;
  startedAt: number;
  progress?: ResearchProgress;
}

/** Tracks the current research run for ChatContext: its task, progress, Stop and failure. */
export function useResearchRun() {
  const [run, setRun] = useState<ResearchRun | null>(null);
  const [failure, setFailure] = useState<ResearchFailure | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const taskIdRef = useRef<string | null>(null);

  const reset = useCallback(() => {
    controllerRef.current = null;
    taskIdRef.current = null;
    setRun(null);
  }, []);

  const begin = useCallback((): ResearchRunOptions => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    taskIdRef.current = null;
    setRun(null);
    setFailure(null);
    // Callbacks from an older run (stopped or replaced) must not change the state.
    const isCurrent = () => controllerRef.current === controller;

    return {
      signal: controller.signal,
      onTaskId: (taskId) => {
        if (!isCurrent()) return;
        taskIdRef.current = taskId;
        setRun({ taskId, startedAt: Date.now() });
      },
      onProgress: (progress) => {
        if (!isCurrent()) return;
        setRun((prev) => (prev ? { ...prev, progress } : prev));
      },
    };
  }, []);

  const stop = useCallback(async () => {
    const controller = controllerRef.current;
    const taskId = taskIdRef.current;
    reset();
    controller?.abort();
    if (!taskId) return;
    try {
      await apiClient.cancelTask(taskId);
    } catch (error) {
      logger.warn("Could not cancel the research task", {
        taskId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }, [reset]);

  const fail = useCallback(
    (kind: ResearchFailureKind, message: string, retry: RetryInfo) => {
      reset();
      setFailure({ kind, message, retry });
    },
    [reset],
  );

  const clear = useCallback(() => {
    reset();
    setFailure(null);
  }, [reset]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  return { run, failure, begin, stop, fail, clear };
}
