"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Modal } from "./Modal";
import { TopicState } from "./TopicState";
import { topicsApi } from "@/utils/topicsApi";
import {
  formatDateTime,
  scheduleInWords,
  skipInWords,
  type TopicSummary,
} from "@/types/topics";

const buttonClass =
  "text-xs py-1 px-2 rounded border border-white/15 text-gray-200 hover:bg-white/10 disabled:opacity-40 disabled:hover:bg-transparent focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 whitespace-nowrap";

/** True while a run is due (next_due has passed) or running, when Run now would only queue a duplicate. */
export function runIsDueOrRunning(topic: TopicSummary, now: Date): boolean {
  if (topic.running) return true;
  if (topic.state !== "active" || !topic.nextDue) return false;
  const due = new Date(topic.nextDue).getTime();
  return Number.isFinite(due) && due <= now.getTime();
}

interface TopicActionsProps {
  topic: TopicSummary;
  /** Called after a successful change, so the page reloads the topic. */
  onChanged: () => void;
  /** Called after a delete; defaults to onChanged. */
  onDeleted?: () => void;
  now?: Date;
}

/** Pause or resume, Run now, and Delete (with a confirmation naming the topic). */
export function TopicActions({
  topic,
  onChanged,
  onDeleted,
  now,
}: TopicActionsProps) {
  const [busy, setBusy] = useState(false);
  const [requested, setRequested] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const act = async (
    action: () => Promise<{ ok: true } | { ok: false; message: string }>,
    done: () => void,
    message?: string,
  ) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    const result = await action();
    setBusy(false);
    if (result.ok) {
      if (message) setNotice(message);
      done();
    } else {
      setError(result.message);
    }
  };

  const paused = topic.state !== "active";
  const runDisabled =
    busy || requested || runIsDueOrRunning(topic, now ?? new Date());
  const label = topic.name;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap gap-1.5">
        {paused ? (
          <button
            type="button"
            className={buttonClass}
            disabled={busy}
            aria-label={`Resume ${label}`}
            onClick={() =>
              act(() => topicsApi.resume(topic.topicId), onChanged)
            }
          >
            Resume
          </button>
        ) : (
          <button
            type="button"
            className={buttonClass}
            disabled={busy}
            aria-label={`Pause ${label}`}
            onClick={() =>
              act(
                () => topicsApi.update(topic.topicId, { status: "paused" }),
                onChanged,
              )
            }
          >
            Pause
          </button>
        )}
        <button
          type="button"
          className={buttonClass}
          disabled={runDisabled}
          aria-label={`Run ${label} now`}
          onClick={() =>
            act(
              () => topicsApi.runNow(topic.topicId),
              () => {
                setRequested(true);
                onChanged();
              },
              "Run requested; it starts within a minute.",
            )
          }
        >
          Run now
        </button>
        <button
          type="button"
          className={`${buttonClass} text-red-200 border-red-500/30 hover:bg-red-900/30`}
          disabled={busy}
          aria-label={`Delete ${label}`}
          onClick={() => setConfirming(true)}
        >
          Delete
        </button>
      </div>
      {notice && (
        <p role="status" className="text-xs text-gray-400">
          {notice}
        </p>
      )}
      {error && !confirming && (
        <p role="alert" className="text-xs text-red-300">
          {error}
        </p>
      )}
      {confirming && (
        <Modal
          titleId={`delete-${topic.topicId}-title`}
          title={`Delete “${topic.name}”?`}
          describedBy={`delete-${topic.topicId}-desc`}
          onClose={() => setConfirming(false)}
        >
          <p
            id={`delete-${topic.topicId}-desc`}
            className="text-sm text-gray-300 mb-6"
          >
            WebAgent stops researching &ldquo;{topic.name}&rdquo; and removes
            its run history and stored reports. This can&apos;t be undone.
          </p>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setConfirming(false)}
              className="px-4 py-2 text-sm text-gray-300 hover:text-white rounded hover:bg-gray-800"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() =>
                act(
                  () => topicsApi.remove(topic.topicId),
                  () => {
                    setConfirming(false);
                    (onDeleted ?? onChanged)();
                  },
                )
              }
              className="px-4 py-2 text-sm bg-red-700 hover:bg-red-600 disabled:opacity-50 text-white rounded"
            >
              Delete topic
            </button>
          </div>
          {error && (
            <p role="alert" className="text-sm text-red-300 mt-3">
              {error}
            </p>
          )}
        </Modal>
      )}
    </div>
  );
}

export function lastRunInWords(topic: TopicSummary): string {
  if (!topic.lastRun?.at) return "Not yet";
  const when = formatDateTime(topic.lastRun.at);
  const n = topic.lastNewSources;
  if (n === undefined) return when;
  return `${when} (${n === 1 ? "1 new source" : `${n} new sources`})`;
}

interface TopicListProps {
  topics: TopicSummary[];
  onChanged: () => void;
  now?: Date;
}

/** The /topics table: name, schedule, state (with skips), next and last run, and actions. */
export function TopicList({ topics, onChanged, now }: TopicListProps) {
  if (topics.length === 0) {
    return (
      <div className="rounded-xl border border-white/10 p-8 text-center text-gray-400">
        <p className="mb-2">No topics yet.</p>
        <p>
          <Link
            href="/"
            className="text-blue-300 underline hover:text-blue-200"
          >
            Track a research answer to see it here.
          </Link>
        </p>
      </div>
    );
  }

  const th =
    "px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-gray-400";
  const td = "px-3 py-3 align-top text-sm";

  return (
    <div className="overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full border-collapse">
        <caption className="sr-only">Tracked topics</caption>
        <thead className="bg-white/5">
          <tr>
            <th scope="col" className={th}>
              Name
            </th>
            <th scope="col" className={th}>
              Schedule
            </th>
            <th scope="col" className={th}>
              State
            </th>
            <th scope="col" className={th}>
              Next run
            </th>
            <th scope="col" className={th}>
              Last run
            </th>
            <th scope="col" className={th}>
              Actions
            </th>
          </tr>
        </thead>
        <tbody>
          {topics.map((topic) => (
            <tr key={topic.topicId} className="border-t border-white/10">
              <td className={td}>
                <Link
                  href={`/topics/${encodeURIComponent(topic.topicId)}`}
                  className="text-gray-100 hover:text-white underline-offset-2 hover:underline"
                >
                  {topic.name}
                </Link>
              </td>
              <td className={`${td} text-gray-300`}>
                {scheduleInWords(topic.schedule)}
              </td>
              <td className={td}>
                <TopicState state={topic.state} />
                {topic.skipped && (
                  <p className="mt-1 text-xs text-amber-200/90">
                    {skipInWords(topic.skipped)}
                  </p>
                )}
              </td>
              <td className={`${td} text-gray-300`}>
                {topic.state === "active" ? formatDateTime(topic.nextDue) : "—"}
              </td>
              <td className={`${td} text-gray-300`}>{lastRunInWords(topic)}</td>
              <td className={td}>
                <TopicActions topic={topic} onChanged={onChanged} now={now} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
