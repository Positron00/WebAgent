"use client";

import React from "react";
import {
  RESEARCH_BUDGET_MESSAGE,
  RESEARCH_STOPPED_MESSAGE,
  RESEARCH_TIMEOUT_MESSAGE,
  RESEARCH_UNREACHABLE_MESSAGE,
  ResearchFailureKind,
} from "@/types/research";

const MAX_REASON_CHARS = 200;

export function failureText(
  kind: ResearchFailureKind,
  message: string,
): string {
  switch (kind) {
    case "timeout":
      return RESEARCH_TIMEOUT_MESSAGE;
    case "budget":
      return RESEARCH_BUDGET_MESSAGE;
    case "unreachable":
      return RESEARCH_UNREACHABLE_MESSAGE;
    case "cancelled":
      return RESEARCH_STOPPED_MESSAGE;
    default: {
      const reason = message.trim().replace(/\.+$/, "") || "unknown error";
      const short =
        reason.length > MAX_REASON_CHARS
          ? `${reason.slice(0, MAX_REASON_CHARS - 1)}…`
          : reason;
      return `Deep research failed: ${short}.`;
    }
  }
}

interface ResearchFailedProps {
  kind: ResearchFailureKind;
  message: string;
  busy: boolean;
  onRetry: () => void;
  onAnswerDirectly: () => void;
}

/** A research run that ended without an answer, with the user's two ways forward. */
export function ResearchFailed({
  kind,
  message,
  busy,
  onRetry,
  onAnswerDirectly,
}: ResearchFailedProps) {
  const buttonClass =
    "text-xs py-1 px-3 rounded transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400";

  return (
    <div className="flex flex-col items-start ml-8 mt-4">
      <div
        className="text-left text-red-300 p-2 bg-red-900/40 rounded w-full mb-2"
        role="alert"
      >
        {failureText(kind, message)}
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={onRetry}
          disabled={busy}
          className={`${buttonClass} bg-blue-800 hover:bg-blue-700 text-white`}
        >
          Try again
        </button>
        <button
          type="button"
          onClick={onAnswerDirectly}
          disabled={busy}
          className={`${buttonClass} border border-white/20 hover:bg-white/10 text-gray-200`}
        >
          Answer directly instead
        </button>
      </div>
    </div>
  );
}
