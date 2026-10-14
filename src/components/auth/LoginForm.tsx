"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { safeNextPath } from "@/utils/safeNext";

export const LOGIN_ERRORS = {
  wrong: "That code is wrong",
  tooMany: "Too many tries, wait a minute",
  config: "Sign-in isn't set up on this server",
  network: "Couldn't reach the server, try again",
};

function errorForStatus(status: number): string {
  if (status === 401) return LOGIN_ERRORS.wrong;
  if (status === 429) return LOGIN_ERRORS.tooMany;
  if (status === 503) return LOGIN_ERRORS.config;
  return LOGIN_ERRORS.network;
}

export default function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const inputRef = useRef<HTMLInputElement>(null);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(
    params.get("error") === "config" ? LOGIN_ERRORS.config : "",
  );

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (busy || !code) return;
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (res.ok) {
        router.replace(safeNextPath(params.get("next")));
        return;
      }
      setError(errorForStatus(res.status));
    } catch {
      setError(LOGIN_ERRORS.network);
    }
    setBusy(false);
    inputRef.current?.select();
  }

  return (
    <form
      onSubmit={onSubmit}
      className="w-full max-w-sm space-y-4 rounded-lg bg-[#232323] p-6 text-gray-100"
    >
      <h1 className="text-xl font-semibold">Sign in to WebAgent</h1>
      <label htmlFor="access-code" className="block text-sm">
        Access code
      </label>
      <input
        id="access-code"
        ref={inputRef}
        type="password"
        autoComplete="current-password"
        value={code}
        onChange={(e) => setCode(e.target.value)}
        maxLength={200}
        className="w-full rounded border border-gray-600 bg-[#191919] px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      {error && (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      )}
      <button
        type="submit"
        disabled={busy || !code}
        className="w-full rounded bg-blue-600 px-3 py-2 font-medium hover:bg-blue-500 disabled:opacity-50"
      >
        {busy ? "Signing in…" : "Sign in"}
      </button>
    </form>
  );
}
