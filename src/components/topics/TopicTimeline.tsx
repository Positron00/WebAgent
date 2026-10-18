"use client";

import React, { useState } from "react";
import { ArticleMarkdown } from "@/components/chat/ArticleMarkdown";
import { SourceCanvas } from "@/components/chat/SourceCanvas";
import { topicsApi } from "@/utils/topicsApi";
import { safeUrl } from "@/utils/safeUrl";
import { CHAT_THEME } from "@/config/chatTheme";
import {
  changeInWords,
  formatDateTime,
  runStatusInWords,
  type TopicRunDetail,
  type TopicRunEntry,
} from "@/types/topics";
import type { Source } from "@/types/chat";

function NewSources({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <ul className="mt-2 space-y-1 text-xs text-gray-300 list-disc pl-5">
      {sources.map((source) => {
        const href = safeUrl(source.url);
        return (
          <li key={source.id}>
            {href ? (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-300 hover:text-blue-200 underline break-all"
              >
                {source.title}
              </a>
            ) : (
              <span className="break-all">{source.title}</span>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function RunReport({ detail }: { detail: TopicRunDetail }) {
  if (!detail.report) {
    return (
      <p className="text-sm text-gray-400">This run has no stored report.</p>
    );
  }
  return (
    <div className="flex flex-col lg:flex-row gap-4 mt-3">
      <div
        className="lg:w-[68%] font-sans antialiased break-words chat-prose chat-answer text-[15px] leading-[1.65] rounded-xl border px-4 py-3"
        style={{
          backgroundColor: CHAT_THEME.pageBackground,
          color: CHAT_THEME.textPrimary,
          borderColor: CHAT_THEME.borderSubtle,
        }}
        role="article"
        aria-label="Stored report"
      >
        <ArticleMarkdown content={detail.report.content} />
      </div>
      {detail.sources.length > 0 && (
        <div className="lg:w-[32%] max-h-[70vh]">
          <SourceCanvas sources={detail.sources} isVisible={true} />
        </div>
      )}
    </div>
  );
}

function TimelineEntry({
  topicId,
  run,
}: {
  topicId: string;
  run: TopicRunEntry;
}) {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<TopicRunDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const when = formatDateTime(run.completedAt ?? run.startedAt);
  const failed = run.status !== "completed" || !!run.errorCode;
  const change = changeInWords(run.change);
  const reportId = `run-${run.runId}-report`;

  const toggle = async () => {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (detail) return;
    setLoading(true);
    setError(null);
    const result = await topicsApi.run(topicId, run.runId);
    setLoading(false);
    if (result.ok && result.value) setDetail(result.value);
    else setError(result.ok ? "The report couldn't be read." : result.message);
  };

  return (
    <li className="border-l-2 border-white/15 pl-4 pb-6 relative">
      <span
        aria-hidden="true"
        className={`absolute -left-[5px] top-1.5 h-2 w-2 rounded-full ${failed ? "bg-red-400" : "bg-emerald-400"}`}
      />
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <time
          dateTime={run.completedAt ?? run.startedAt}
          className="text-sm text-gray-200"
        >
          {when}
        </time>
        <span
          className={`text-sm ${failed ? "text-red-300" : "text-gray-300"}`}
        >
          {runStatusInWords(run.status, run.errorCode)}
        </span>
        {!failed && change && (
          <span className="text-sm text-gray-400">{change}</span>
        )}
      </div>
      {!failed && run.change && (
        <NewSources sources={run.change.newSourceList} />
      )}
      {run.status === "completed" && (
        <button
          type="button"
          onClick={toggle}
          aria-expanded={open}
          aria-controls={reportId}
          className="mt-2 text-xs py-1 px-2 rounded border border-white/15 text-gray-200 hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
        >
          {open ? "Hide report" : "Open report"}
        </button>
      )}
      {open && (
        <div id={reportId}>
          {loading && (
            <p role="status" className="text-sm text-gray-400 mt-2">
              Loading the report…
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm text-red-300 mt-2">
              {error}
            </p>
          )}
          {detail && <RunReport detail={detail} />}
        </div>
      )}
    </li>
  );
}

/** A topic's runs, newest first, each with its change and its stored report (REC-116). */
export function TopicTimeline({
  topicId,
  runs,
}: {
  topicId: string;
  runs: TopicRunEntry[];
}) {
  if (runs.length === 0) {
    return (
      <p className="text-sm text-gray-400">
        No runs yet. The first one runs at the next scheduled time.
      </p>
    );
  }
  const time = (run: TopicRunEntry) => {
    const t = new Date(run.startedAt ?? run.completedAt ?? "").getTime();
    return Number.isFinite(t) ? t : 0;
  };
  const newestFirst = [...runs].sort((a, b) => time(b) - time(a));
  return (
    <ol aria-label="Runs, newest first" className="ml-1">
      {newestFirst.map((run) => (
        <TimelineEntry key={run.runId} topicId={topicId} run={run} />
      ))}
    </ol>
  );
}
