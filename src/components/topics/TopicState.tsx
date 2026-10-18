import React from "react";
import type { TopicState as TopicStateValue } from "@/types/topics";

export const TOPIC_STATE_LABELS: Record<TopicStateValue, string> = {
  active: "Active",
  paused: "Paused",
  held_failures: "Held: failures",
  held_budget: "Held: budget",
};

const STYLES: Record<TopicStateValue, string> = {
  active: "border-emerald-500/40 bg-emerald-950/40 text-emerald-200",
  paused: "border-gray-500/40 bg-gray-800/60 text-gray-300",
  held_failures: "border-red-500/40 bg-red-950/40 text-red-200",
  held_budget: "border-amber-500/40 bg-amber-950/40 text-amber-200",
};

const ICONS: Record<TopicStateValue, string> = {
  active: "●",
  paused: "❚❚",
  held_failures: "!",
  held_budget: "$",
};

/** The effective state as a chip: the words carry the meaning, the colour only repeats it. */
export function TopicState({ state }: { state: TopicStateValue }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs whitespace-nowrap ${STYLES[state]}`}
      data-state={state}
    >
      <span aria-hidden="true" className="text-[10px]">
        {ICONS[state]}
      </span>
      <span>{TOPIC_STATE_LABELS[state]}</span>
    </span>
  );
}
