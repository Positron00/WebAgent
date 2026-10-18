"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { TopicList } from "@/components/topics/TopicList";
import { topicsApi } from "@/utils/topicsApi";
import { useVisibleInterval } from "@/hooks/useVisibleInterval";
import type { TopicSummary } from "@/types/topics";

const REFRESH_MS = 30_000;

export default function TopicsPage() {
  const [topics, setTopics] = useState<TopicSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const result = await topicsApi.list();
    if (result.ok) {
      setTopics(result.value);
      setError(null);
    } else {
      setError(result.message);
    }
  }, []);

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
        <nav className="mb-4 text-sm">
          <Link href="/" className="text-gray-400 hover:text-white">
            ← Back to chat
          </Link>
        </nav>
        <h1 className="text-2xl font-semibold text-[#c4c4c4] mb-1">Topics</h1>
        <p className="text-sm text-gray-400 mb-6">
          Questions WebAgent researches again on a schedule. This list refreshes
          every 30 seconds.
        </p>
        {error && (
          <p
            role="alert"
            className="mb-4 text-sm text-red-300 bg-red-900/40 rounded px-3 py-2"
          >
            {error}
          </p>
        )}
        {topics === null ? (
          !error && (
            <p role="status" className="text-sm text-gray-400">
              Loading topics…
            </p>
          )
        ) : (
          <TopicList topics={topics} onChanged={() => void load()} />
        )}
      </div>
    </main>
  );
}
