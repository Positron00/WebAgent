"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { TopicActions } from "@/components/topics/TopicList";
import { TopicState } from "@/components/topics/TopicState";
import { TopicTimeline } from "@/components/topics/TopicTimeline";
import { SubscriberList } from "@/components/topics/SubscriberList";
import { topicsApi } from "@/utils/topicsApi";
import { useVisibleInterval } from "@/hooks/useVisibleInterval";
import {
  formatDateTime,
  scheduleInWords,
  skipInWords,
  type TopicRunEntry,
  type TopicSummary,
} from "@/types/topics";

const REFRESH_MS = 30_000;

export default function TopicDetailPage() {
  const params = useParams<{ topicId: string }>();
  const topicId = typeof params?.topicId === "string" ? params.topicId : "";
  const router = useRouter();
  const [topic, setTopic] = useState<TopicSummary | null>(null);
  const [runs, setRuns] = useState<TopicRunEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    if (!topicId) return;
    // The single-topic reply is the definition; the list item adds the effective state, next
    // and last run and skips, so the page prefers the list's view of this topic.
    const [one, history, list] = await Promise.all([
      topicsApi.get(topicId),
      topicsApi.runs(topicId),
      topicsApi.list(),
    ]);
    setLoaded(true);
    if (!one.ok) {
      setError(one.message);
      if (one.status === 404) setTopic(null);
      return;
    }
    const fromList = list.ok
      ? list.value.find((t) => t.topicId === topicId)
      : undefined;
    // Only the definition lists the subscribers, so they're taken from it.
    setTopic(
      fromList
        ? { ...fromList, subscribers: one.value?.subscribers }
        : one.value,
    );
    setError(one.value || fromList ? null : "This topic couldn't be read.");
    if (history.ok) setRuns(history.value);
  }, [topicId]);

  useEffect(() => {
    void load();
  }, [load]);
  useVisibleInterval(() => void load(), REFRESH_MS);

  return (
    <main
      className="min-h-screen font-sans antialiased px-4 py-8"
      style={{ backgroundColor: "#191919", color: "#e5e5e5" }}
    >
      <div className="mx-auto w-full max-w-5xl">
        <nav className="mb-4 text-sm flex gap-4">
          <Link href="/topics" className="text-gray-400 hover:text-white">
            ← All topics
          </Link>
          <Link href="/" className="text-gray-400 hover:text-white">
            Chat
          </Link>
        </nav>
        {error && (
          <p
            role="alert"
            className="mb-4 text-sm text-red-300 bg-red-900/40 rounded px-3 py-2"
          >
            {error}
          </p>
        )}
        {!loaded && (
          <p role="status" className="text-sm text-gray-400">
            Loading the topic…
          </p>
        )}
        {topic && (
          <>
            <header className="mb-8 space-y-2">
              <div className="flex flex-wrap items-center gap-3">
                <h1 className="text-2xl font-semibold text-[#c4c4c4]">
                  {topic.name}
                </h1>
                <TopicState state={topic.state} />
              </div>
              <p className="text-sm text-gray-300 whitespace-pre-wrap">
                {topic.query}
              </p>
              <p className="text-sm text-gray-400">
                {scheduleInWords(topic.schedule)}
                {topic.state === "active" && topic.nextDue
                  ? ` · next run ${formatDateTime(topic.nextDue)}`
                  : ""}
              </p>
              {topic.skipped && (
                <p className="text-sm text-amber-200/90">
                  {skipInWords(topic.skipped)}
                </p>
              )}
              <TopicActions
                topic={topic}
                onChanged={() => void load()}
                onDeleted={() => router.push("/topics")}
              />
            </header>
            <SubscriberList
              topicId={topic.topicId}
              subscribers={topic.subscribers ?? []}
              onChanged={() => void load()}
            />
            <h2 className="text-lg font-medium text-gray-200 mb-4">Runs</h2>
            <TopicTimeline topicId={topic.topicId} runs={runs} />
          </>
        )}
      </div>
    </main>
  );
}
