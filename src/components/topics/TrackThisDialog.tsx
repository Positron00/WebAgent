"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Modal } from "./Modal";
import { topicsApi } from "@/utils/topicsApi";
import {
  WEEKDAYS,
  WEEKDAY_NAMES,
  type NotifyWhen,
  type Weekday,
} from "@/types/topics";

export const TOPIC_NAME_MAX = 80;
export const TOPIC_QUERY_MAX = 2000;
const JS_DAY_TO_WEEKDAY: Weekday[] = [
  "sun",
  "mon",
  "tue",
  "wed",
  "thu",
  "fri",
  "sat",
];

export function browserTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

interface TrackThisDialogProps {
  taskId: string;
  /** `result.standalone_query`, or the user's question when the backend gave none. */
  query: string;
  /** The report's title; cut to 80 characters for the name. */
  title?: string;
  onClose: () => void;
  /** Called after the topic is created (tests and the parent may use it). */
  onCreated?: () => void;
  /** For tests: today's date. */
  now?: Date;
}

/** The "Track this" form: turns a research answer into a recurring topic (REC-116). */
export function TrackThisDialog({
  taskId,
  query: initialQuery,
  title,
  onClose,
  onCreated,
  now,
}: TrackThisDialogProps) {
  const timeZone = browserTimeZone();
  const [name, setName] = useState(
    (title?.trim() || initialQuery.trim()).slice(0, TOPIC_NAME_MAX),
  );
  const [query, setQuery] = useState(initialQuery.slice(0, TOPIC_QUERY_MAX));
  const [every, setEvery] = useState<"day" | "week">("week");
  const [weekday, setWeekday] = useState<Weekday>(
    JS_DAY_TO_WEEKDAY[(now ?? new Date()).getDay()],
  );
  const [time, setTime] = useState("07:30");
  const [notifyWhen, setNotifyWhen] = useState<NotifyWhen>("changes");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState(false);

  const canSubmit =
    !busy &&
    name.trim().length > 0 &&
    query.trim().length > 0 &&
    /^\d{2}:\d{2}$/.test(time);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    const result = await topicsApi.create({
      name: name.trim(),
      query: query.trim(),
      schedule: {
        every,
        ...(every === "week" ? { weekday } : {}),
        time,
        tz: timeZone,
      },
      notify: { when: notifyWhen },
      created_from_task: taskId,
    });
    setBusy(false);
    if (result.ok) {
      setCreated(true);
      onCreated?.();
    } else {
      setError(result.message);
    }
  };

  const inputClass =
    "w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-sm text-gray-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400";
  const labelClass = "block text-xs font-medium text-gray-300 mb-1";

  if (created) {
    return (
      <Modal titleId="track-this-title" title="Topic created" onClose={onClose}>
        <p className="text-sm text-gray-300 mb-5" role="status">
          WebAgent will research &ldquo;{name.trim()}&rdquo; on schedule. You
          can follow it in{" "}
          <Link
            href="/topics"
            className="text-blue-300 underline hover:text-blue-200"
          >
            Topics
          </Link>
          .
        </p>
        <div className="flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded"
          >
            Close
          </button>
        </div>
      </Modal>
    );
  }

  return (
    <Modal
      titleId="track-this-title"
      title="Track this question"
      onClose={onClose}
    >
      <form onSubmit={submit} className="space-y-3" noValidate>
        <div>
          <label htmlFor="track-name" className={labelClass}>
            Name
          </label>
          <input
            id="track-name"
            className={inputClass}
            value={name}
            maxLength={TOPIC_NAME_MAX}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>
        <div>
          <label htmlFor="track-query" className={labelClass}>
            Question
          </label>
          <textarea
            id="track-query"
            className={`${inputClass} min-h-[4.5rem]`}
            value={query}
            maxLength={TOPIC_QUERY_MAX}
            onChange={(e) => setQuery(e.target.value)}
            required
          />
        </div>
        <div className="flex gap-3">
          <div className="flex-1">
            <label htmlFor="track-every" className={labelClass}>
              Frequency
            </label>
            <select
              id="track-every"
              className={inputClass}
              value={every}
              onChange={(e) =>
                setEvery(e.target.value === "day" ? "day" : "week")
              }
            >
              <option value="day">Daily</option>
              <option value="week">Weekly</option>
            </select>
          </div>
          {every === "week" && (
            <div className="flex-1">
              <label htmlFor="track-weekday" className={labelClass}>
                Weekday
              </label>
              <select
                id="track-weekday"
                className={inputClass}
                value={weekday}
                onChange={(e) => setWeekday(e.target.value as Weekday)}
              >
                {WEEKDAYS.map((day) => (
                  <option key={day} value={day}>
                    {WEEKDAY_NAMES[day]}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="flex-1">
            <label htmlFor="track-time" className={labelClass}>
              Time
            </label>
            <input
              id="track-time"
              type="time"
              className={inputClass}
              value={time}
              onChange={(e) => setTime(e.target.value)}
              required
            />
          </div>
        </div>
        <div>
          <label htmlFor="track-tz" className={labelClass}>
            Time zone
          </label>
          <input
            id="track-tz"
            className={`${inputClass} text-gray-400`}
            value={timeZone}
            readOnly
            aria-readonly="true"
          />
        </div>
        <fieldset>
          <legend className={labelClass}>Notify when</legend>
          <div className="flex gap-4 text-sm text-gray-200">
            <label className="flex items-center gap-1.5">
              <input
                type="radio"
                name="track-notify"
                value="changes"
                checked={notifyWhen === "changes"}
                onChange={() => setNotifyWhen("changes")}
              />
              Something changes
            </label>
            <label className="flex items-center gap-1.5">
              <input
                type="radio"
                name="track-notify"
                value="always"
                checked={notifyWhen === "always"}
                onChange={() => setNotifyWhen("always")}
              />
              Every run
            </label>
          </div>
        </fieldset>
        {error && (
          <p
            role="alert"
            className="text-sm text-red-300 bg-red-900/40 rounded px-2 py-1.5"
          >
            {error}
          </p>
        )}
        <div className="flex justify-end gap-2 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm text-gray-300 hover:text-white rounded hover:bg-gray-800"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={!canSubmit}
            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded"
          >
            {busy ? "Creating…" : "Create topic"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
