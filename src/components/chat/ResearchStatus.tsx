"use client";

import React, { useEffect, useState } from "react";
import { ResearchProgress } from "@/types/research";

interface ResearchStatusProps {
  progress?: ResearchProgress;
  /** When the research task started (ms since epoch); the elapsed time counts from here. */
  startedAt: number;
  reducedMotion: boolean;
  onStop: () => void;
}

/** 75 000 ms -> "1:15" */
export function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/** "Thinking…" for a research run: current step, latest log line, elapsed time and Stop. */
export function ResearchStatus({
  progress,
  startedAt,
  reducedMotion,
  onStop,
}: ResearchStatusProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const step = progress?.step || "Working";

  return (
    <div className="flex items-start gap-3 ml-8 mt-4 text-[#8e8e8e] text-[15px]">
      <div
        className={`${
          reducedMotion ? "" : "animate-spin motion-reduce:animate-none"
        } mt-1 shrink-0 rounded-full h-4 w-4 border-2 border-gray-300 border-t-transparent`}
        aria-hidden="true"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-3">
          {/* Only the step is announced; the clock would be read out every second. */}
          <span role="status">Thinking… · {step}</span>
          <span className="text-xs tabular-nums">
            {formatElapsed(now - startedAt)}
          </span>
        </div>
        {progress?.latestLog && (
          <p
            className="mt-1 text-xs text-gray-500 truncate"
            title={progress.latestLog}
          >
            {progress.latestLog}
          </p>
        )}
      </div>
      <button
        type="button"
        onClick={onStop}
        aria-label="Stop research"
        className="shrink-0 text-xs border border-white/20 hover:bg-white/10 text-gray-200 py-1 px-3 rounded transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
      >
        Stop
      </button>
    </div>
  );
}
