"use client";

import React, { useState } from "react";
import { topicsApi } from "@/utils/topicsApi";
import {
  formatDate,
  isPlausibleEmail,
  INVALID_EMAIL_MESSAGE,
  MAX_SUBSCRIBERS,
  type TopicSubscriber,
} from "@/types/topics";

export const CONFIRMATION_HINT = "We'll send a confirmation link first.";

const buttonClass =
  "text-xs py-1 px-2 rounded border border-white/15 text-gray-200 hover:bg-white/10 disabled:opacity-40 disabled:hover:bg-transparent focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 whitespace-nowrap";

interface SubscriberListProps {
  topicId: string;
  subscribers: TopicSubscriber[];
  /** Called after an add or remove, so the page reloads the topic. */
  onChanged: () => void;
}

/** The topic's email subscribers (REC-133): address, confirmed or pending, add and remove. */
export function SubscriberList({
  topicId,
  subscribers,
  onChanged,
}: SubscriberListProps) {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const full = subscribers.length >= MAX_SUBSCRIBERS;

  const run = async (
    action: () => ReturnType<typeof topicsApi.addSubscriber>,
    message: string,
    after?: () => void,
  ) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    const result = await action();
    setBusy(false);
    if (result.ok) {
      setNotice(message);
      after?.();
      onChanged();
    } else {
      setError(result.message);
    }
  };

  const add = (event: React.FormEvent) => {
    event.preventDefault();
    const address = email.trim();
    if (!isPlausibleEmail(address)) {
      setNotice(null);
      setError(INVALID_EMAIL_MESSAGE);
      return;
    }
    void run(
      () => topicsApi.addSubscriber(topicId, address),
      `A confirmation link was sent to ${address}.`,
      () => setEmail(""),
    );
  };

  return (
    <section aria-labelledby="subscribers-title" className="mb-8 space-y-3">
      <h2 id="subscribers-title" className="text-lg font-medium text-gray-200">
        Email
      </h2>
      {subscribers.length === 0 ? (
        <p className="text-sm text-gray-400">Nobody gets emails yet.</p>
      ) : (
        <ul className="space-y-1.5" aria-label="Subscribers">
          {subscribers.map((s) => (
            <li
              key={s.email}
              className="flex flex-wrap items-center gap-3 text-sm"
            >
              <span className="text-gray-100 break-all">{s.email}</span>
              <span
                className={
                  s.confirmed
                    ? "text-xs text-green-300"
                    : "text-xs text-amber-200/90"
                }
              >
                {s.confirmed
                  ? "Confirmed"
                  : `Waiting for confirmation${s.addedAt ? ` (since ${formatDate(s.addedAt)})` : ""}`}
              </span>
              <button
                type="button"
                className={buttonClass}
                disabled={busy}
                aria-label={`Remove ${s.email}`}
                onClick={() =>
                  void run(
                    () => topicsApi.removeSubscriber(topicId, s.email),
                    `${s.email} won't get more emails.`,
                  )
                }
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
      <form
        onSubmit={add}
        className="flex flex-wrap items-end gap-2"
        noValidate
      >
        <div>
          <label
            htmlFor="subscriber-email"
            className="block text-xs font-medium text-gray-300 mb-1"
          >
            Add an email address
          </label>
          <input
            id="subscriber-email"
            type="email"
            autoComplete="email"
            className="w-72 max-w-full rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-sm text-gray-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
            value={email}
            maxLength={254}
            disabled={full}
            aria-describedby="subscriber-hint"
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <button
          type="submit"
          className={`${buttonClass} py-1.5`}
          disabled={busy || full || !email.trim()}
        >
          Add
        </button>
      </form>
      <p id="subscriber-hint" className="text-xs text-gray-400">
        {full
          ? `A topic can have at most ${MAX_SUBSCRIBERS} addresses.`
          : CONFIRMATION_HINT}
      </p>
      {error && (
        <p
          role="alert"
          className="text-sm text-red-300 bg-red-900/40 rounded px-2 py-1.5"
        >
          {error}
        </p>
      )}
      {notice && (
        <p role="status" className="text-sm text-gray-300">
          {notice}
        </p>
      )}
    </section>
  );
}
