"use client";

import React, { useRef, useEffect, useState } from "react";
import { ChatMessage } from "@/types/chat";
import { useApp } from "@/contexts/AppContext";
import { useChat } from "@/contexts/ChatContext";
import { RouteLabel } from "./RouteLabel";
import { SecuritySummary } from "./SecuritySummary";
import { ResearchStatus } from "./ResearchStatus";
import { ResearchFailed } from "./ResearchFailed";
import {
  TrackThisDialog,
  TOPIC_NAME_MAX,
} from "@/components/topics/TrackThisDialog";
import ReactMarkdown from "react-markdown";
import { ArticleMarkdown } from "./ArticleMarkdown";
import { InformationCircleIcon, BeakerIcon } from "@heroicons/react/24/outline";
import { CHAT_THEME } from "@/config/chatTheme";

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
  error?: string;
}

// Function to check if a message contains code blocks
export const hasCodeBlocks = (content: string): boolean => {
  // Check for markdown code blocks (```code```)
  const codeBlockRegex = /```[\s\S]*?```/g;
  return codeBlockRegex.test(content);
};

// Function to extract code blocks from a message
export const extractCodeBlocks = (
  content: string,
): { language: string; code: string }[] => {
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
  const codeBlocks: { language: string; code: string }[] = [];

  let match;
  while ((match = codeBlockRegex.exec(content)) !== null) {
    codeBlocks.push({
      language: match[1] || "text",
      code: match[2].trim(),
    });
  }

  return codeBlocks;
};

// Add academic-style citation references to the text
const addCitationReferences = (
  content: string,
  sourcesCount: number,
): string => {
  if (sourcesCount === 0) return content;

  // Create array of available citation numbers
  const citationNumbers = Array.from({ length: sourcesCount }, (_, i) => i + 1);
  let modifiedContent = content;

  // Find sentences to cite
  const sentences = modifiedContent.match(/[^.!?]+[.!?]+/g) || [];

  if (sentences.length === 0) {
    // If we couldn't parse sentences, just add a citation at the end
    return modifiedContent + ` [${citationNumbers[0]}]`;
  }

  // Always cite at least 3 sentences if possible, or all sentences if fewer than 3
  const sentencesToCite = Math.min(
    Math.max(3, Math.ceil(sentences.length * 0.4)),
    sentences.length,
  );

  // Select sentences to cite, prioritizing the final sentence
  // and then evenly distributed throughout the text
  const selectedIndices = new Set<number>();

  // Always cite the last sentence if available
  if (sentences.length > 0) {
    selectedIndices.add(sentences.length - 1);
  }

  // Distribute remaining citations throughout the text
  while (
    selectedIndices.size < sentencesToCite &&
    selectedIndices.size < citationNumbers.length
  ) {
    // Find an index that hasn't been selected yet
    const index = Math.floor(Math.random() * sentences.length);
    if (!selectedIndices.has(index)) {
      selectedIndices.add(index);
    }
  }

  // Apply citations
  const indices = Array.from(selectedIndices).sort((a, b) => a - b);
  for (let i = 0; i < indices.length && i < citationNumbers.length; i++) {
    const sentenceIndex = indices[i];
    const citation = citationNumbers[i];

    // Skip sentences that already have citations
    if (sentences[sentenceIndex] && !sentences[sentenceIndex].includes("[")) {
      const citedSentence = sentences[sentenceIndex].replace(
        /([.!?]+)$/,
        ` [${citation}]$1`,
      );

      // Escape special regex characters in the original sentence
      const escapeRegex = (string: string): string =>
        string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const originalSentenceEscaped = escapeRegex(sentences[sentenceIndex]);

      // Replace just this occurrence
      modifiedContent = modifiedContent.replace(
        new RegExp(`${originalSentenceEscaped}(?![^.!?]*\\[)`),
        citedSentence,
      );

      // Update our record of the sentence
      sentences[sentenceIndex] = citedSentence;
    }
  }

  // If we somehow didn't add any citations, add one to the last sentence
  if (!modifiedContent.includes("[")) {
    modifiedContent = modifiedContent.replace(
      /([.!?]+)$/,
      ` [${citationNumbers[0]}]$1`,
    );
  }

  return modifiedContent;
};

type ConversationTurn = {
  user: ChatMessage;
  assistant?: ChatMessage;
};

function groupMessagesIntoTurns(messages: ChatMessage[]): ConversationTurn[] {
  const turns: ConversationTurn[] = [];
  let pending: ConversationTurn | null = null;

  for (const message of messages) {
    if (message.role === "user") {
      if (pending) {
        turns.push(pending);
      }
      pending = { user: message };
      continue;
    }
    if (message.role === "assistant") {
      if (pending) {
        pending.assistant = message;
        turns.push(pending);
        pending = null;
      } else if (turns.length > 0) {
        // A second answer to the same question ("Research this in depth") gets its own turn.
        turns.push({ user: turns[turns.length - 1].user, assistant: message });
      }
    }
  }

  if (pending) {
    turns.push(pending);
  }

  return turns;
}

/** Checks only that the web app answers (GET /api/ping); it never sends a chat request. */
function DiagnosticsCheck() {
  const [result, setResult] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  const runCheck = async () => {
    setChecking(true);
    try {
      const response = await fetch("/api/ping", { cache: "no-store" });
      setResult(
        response.ok
          ? "Web app reachable"
          : `Web app answered HTTP ${response.status}`,
      );
    } catch {
      setResult("Web app not reachable");
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={runCheck}
        disabled={checking}
        className="text-xs bg-blue-800 hover:bg-blue-700 disabled:opacity-50 text-white py-1 px-2 rounded transition-colors text-left"
      >
        Run Diagnostics
      </button>
      <span role="status" className="text-xs text-gray-400">
        {result}
      </span>
    </div>
  );
}

export function MessageList({ messages, isLoading, error }: MessageListProps) {
  const { accessibility, agenticMode } = useApp();
  const {
    researchInDepth,
    approveAndRerun,
    researchRun,
    researchFailure,
    stopResearch,
    retryResearch,
    answerDirectlyInstead,
  } = useChat();
  const latestTurnRef = useRef<HTMLDivElement>(null);
  const [tracking, setTracking] = useState<{
    taskId: string;
    query: string;
    title?: string;
  } | null>(null);
  const turns = groupMessagesIntoTurns(messages);

  // Keep the latest question visible at the top of the scroll area (Perplexity-style)
  useEffect(() => {
    if (!latestTurnRef.current || turns.length === 0) return;
    latestTurnRef.current.scrollIntoView({
      block: "start",
      behavior: accessibility.reducedMotion ? "auto" : "smooth",
    });
  }, [turns.length, isLoading, accessibility.reducedMotion]);

  // Apply custom colors or high contrast theme if enabled
  const getMessageClassName = (role: string) => {
    const sans =
      "font-sans antialiased break-words max-w-3xl w-full mx-auto text-[15px] leading-[1.65]";
    if (role === "user") {
      return `${sans} py-3.5 px-5 rounded-2xl text-center shrink-0`;
    }
    return `font-sans antialiased break-words w-full max-w-none py-4 px-1 text-left chat-prose chat-answer text-[15px] leading-[1.65]`;
  };

  // Get styles for messages based on accessibility settings
  const getMessageStyles = (role: string) => {
    // If high contrast is enabled, use those colors instead of custom colors
    if (accessibility.highContrast) {
      if (role === "user") {
        return {
          backgroundColor: "#1a1a1a",
          color: CHAT_THEME.textPrimary,
        };
      }
      return {
        backgroundColor: CHAT_THEME.pageBackground,
        color: CHAT_THEME.textPrimary,
      };
    }

    // Otherwise use custom colors
    if (role === "user") {
      return {
        backgroundColor: accessibility.queryBackgroundColor,
        color: accessibility.queryTextColor,
      };
    } else {
      // For assistant messages
      return {
        backgroundColor: accessibility.responseBackgroundColor,
        color: accessibility.responseTextColor,
      };
    }
  };

  // Render article-formatted assistant messages
  const renderArticleContent = (content: string, hasSources: boolean) => {
    return (
      <div className="relative">
        <ArticleMarkdown content={content} />

        {hasSources && accessibility.citeSources && (
          <div className="absolute top-0 right-0 text-[#8e8e8e] text-xs px-2 py-1 rounded-full border border-white/10 flex items-center">
            <InformationCircleIcon className="w-3 h-3 mr-1" />
            <span>Cited Sources</span>
          </div>
        )}
      </div>
    );
  };

  // Render workflow logs
  const renderWorkflowLogs = (logs?: string[], routeAgentic?: boolean) => {
    if (!logs || logs.length === 0 || !routeAgentic) {
      return null;
    }

    return (
      <div className="mt-3 text-gray-500 text-xs border-t border-gray-800 pt-2">
        <div className="flex items-center mb-1">
          <BeakerIcon className="w-3 h-3 mr-1" />
          <span className="text-gray-400 font-semibold">Workflow Logs</span>
        </div>
        <div className="text-left font-mono pl-1">
          {logs.map((log, index) => (
            <div key={index} className="mb-1 whitespace-pre-wrap opacity-75">
              {log}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full text-left mb-4">
      <div
        className="w-full flex-1 min-h-[50vh] max-h-[70vh] rounded-xl border overflow-y-auto px-2 py-3 font-sans"
        style={{
          backgroundColor: CHAT_THEME.pageBackground,
          color: CHAT_THEME.textPrimary,
          borderColor: CHAT_THEME.borderSubtle,
        }}
        role="log"
        aria-live="polite"
        aria-label="Chat Messages"
      >
        {messages.length === 0 ? (
          <div
            className="h-full flex flex-col items-center justify-center p-8 text-[15px]"
            style={{ color: CHAT_THEME.textMuted }}
          >
            <span>Start a conversation...</span>
            <span>Responses will appear here</span>
          </div>
        ) : (
          <>
            {turns.map((turn, index) => {
              const { user: userMessage, assistant: assistantMessage } = turn;
              const isLatestTurn = index === turns.length - 1;

              let processedContent = assistantMessage?.content;
              const hasSources = !!(
                assistantMessage?.sources && assistantMessage.sources.length > 0
              );
              if (
                hasSources &&
                assistantMessage &&
                assistantMessage.sources &&
                accessibility.citeSources
              ) {
                processedContent = addCitationReferences(
                  assistantMessage.content,
                  assistantMessage.sources.length,
                );
              }

              return (
                <div
                  key={assistantMessage?.id ?? userMessage.id}
                  ref={isLatestTurn ? latestTurnRef : undefined}
                  className="mb-8 pb-8 border-b last:border-b-0 last:mb-0 last:pb-0"
                  style={{ borderColor: CHAT_THEME.borderSubtle }}
                >
                  <div className="mb-5 flex justify-center">
                    <div
                      className={getMessageClassName("user")}
                      role="note"
                      aria-label="user's message"
                      style={getMessageStyles("user")}
                    >
                      <div className="chat-prose chat-question text-center [&_p]:text-center [&_p]:mb-0">
                        <ReactMarkdown>{userMessage.content}</ReactMarkdown>
                      </div>
                    </div>
                  </div>

                  {assistantMessage && (
                    <div className="relative w-full text-left">
                      <div
                        className={getMessageClassName("assistant")}
                        role="article"
                        aria-label="assistant's response"
                        style={getMessageStyles("assistant")}
                      >
                        {assistantMessage.refusal ? (
                          <p className="text-sm text-amber-200/90 border border-amber-500/30 bg-amber-950/30 rounded-lg px-3 py-2">
                            {assistantMessage.refusal.code === "escalated"
                              ? "Escalated"
                              : "Declined"}
                            : {assistantMessage.refusal.reason}
                          </p>
                        ) : (
                          <>
                            {processedContent &&
                              renderArticleContent(
                                processedContent,
                                hasSources,
                              )}
                            {assistantMessage.route && (
                              <RouteLabel
                                info={assistantMessage.route}
                                mode={agenticMode}
                                canOverride={
                                  !userMessage.imageAttached && !isLoading
                                }
                                onOverride={() => {
                                  if (assistantMessage.id) {
                                    researchInDepth(assistantMessage.id);
                                  }
                                }}
                              />
                            )}
                            {assistantMessage.route?.route === "agentic" &&
                              assistantMessage.task?.id && (
                                <div className="mt-2">
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setTracking({
                                        taskId: assistantMessage.task!.id,
                                        query:
                                          assistantMessage.task!
                                            .standaloneQuery ||
                                          userMessage.content,
                                        title:
                                          assistantMessage.task!.title?.slice(
                                            0,
                                            TOPIC_NAME_MAX,
                                          ),
                                      })
                                    }
                                    className="text-xs py-1 px-2 rounded border border-white/15 text-gray-300 hover:text-white hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
                                    aria-label="Track this question on a schedule"
                                  >
                                    Track this
                                  </button>
                                </div>
                              )}
                            {assistantMessage.security &&
                              assistantMessage.id && (
                                <SecuritySummary
                                  security={assistantMessage.security}
                                  busy={isLoading}
                                  onApprove={() =>
                                    approveAndRerun(assistantMessage.id!)
                                  }
                                />
                              )}
                            {renderWorkflowLogs(
                              assistantMessage.logs,
                              assistantMessage.route?.route === "agentic",
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </>
        )}
        {isLoading && researchRun ? (
          <ResearchStatus
            progress={researchRun.progress}
            startedAt={researchRun.startedAt}
            reducedMotion={accessibility.reducedMotion}
            onStop={stopResearch}
          />
        ) : (
          isLoading && (
            <div
              className="flex items-center justify-start space-x-2 ml-8 mt-4 text-[#8e8e8e] text-[15px]"
              role="status"
              aria-label="Loading response"
            >
              <div
                className={`${
                  accessibility.reducedMotion
                    ? ""
                    : "animate-spin motion-reduce:animate-none"
                } rounded-full h-4 w-4 border-2 border-gray-300 border-t-transparent`}
                aria-hidden="true"
              />
              <span>Thinking...</span>
            </div>
          )
        )}
        {researchFailure ? (
          <ResearchFailed
            kind={researchFailure.kind}
            message={researchFailure.message}
            busy={isLoading}
            onRetry={retryResearch}
            onAnswerDirectly={answerDirectlyInstead}
          />
        ) : (
          error && (
            <div className="flex flex-col items-start ml-8 mt-4">
              <div
                className="text-left text-red-400 p-2 bg-red-900/50 rounded w-full mb-2"
                role="alert"
                aria-live="assertive"
              >
                {error}
              </div>
              <DiagnosticsCheck />
            </div>
          )
        )}
      </div>
      {tracking && (
        <TrackThisDialog
          taskId={tracking.taskId}
          query={tracking.query}
          title={tracking.title}
          onClose={() => setTracking(null)}
        />
      )}
    </div>
  );
}
