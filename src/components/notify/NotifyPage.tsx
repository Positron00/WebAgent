import React from "react";
import {
  lookupNotifyToken,
  verifyReceipt,
  type NotifyPurpose,
} from "@/app/api/notify/shared";

// The public confirm and unsubscribe pages (REC-134), as a server component. Opening a page
// changes nothing: it only reads the topic name through the backend's read-only lookup. The
// button's form POSTs the token to /api/notify/<purpose>, which redirects back here with a
// signed receipt (or an error) for the result view.

export type NotifySearchParams = Promise<
  Record<string, string | string[] | undefined>
>;

export const INVALID_LINK = "This link has expired or isn't valid.";
export const UNAVAILABLE =
  "This can't be done right now. Please try the link again later.";

const COPY: Record<
  NotifyPurpose,
  {
    heading: (name: string) => string;
    body: string;
    button: string;
    done: (name: string) => string;
  }
> = {
  confirm: {
    heading: (name) => `Get emails about “${name}”?`,
    body: "Press the button to confirm this address. Nothing is sent until you do.",
    button: "Confirm",
    done: (name) => `Confirmed: you'll get emails about “${name}”.`,
  },
  unsubscribe: {
    heading: (name) => `Stop emails about “${name}”?`,
    body: "Press the button and this address won't get more emails about this topic.",
    button: "Unsubscribe",
    done: (name) => `You won't get more emails about “${name}”.`,
  },
};

const first = (v: string | string[] | undefined) =>
  Array.isArray(v) ? v[0] : v;

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <main
      className="min-h-screen flex items-center justify-center px-4 py-8 font-sans antialiased"
      style={{ backgroundColor: "#191919", color: "#e5e5e5" }}
    >
      <div className="w-full max-w-md rounded-lg border border-gray-800 bg-[#202020] p-6 space-y-4">
        <p className="text-xs uppercase tracking-wide text-gray-500">
          WebAgent
        </p>
        {children}
      </div>
    </main>
  );
}

function Message({ text, role }: { text: string; role: "status" | "alert" }) {
  return (
    <Shell>
      <p role={role} className="text-base text-gray-100">
        {text}
      </p>
    </Shell>
  );
}

export async function NotifyPage({
  purpose,
  searchParams,
}: {
  purpose: NotifyPurpose;
  searchParams: NotifySearchParams;
}) {
  const params = await searchParams;
  const copy = COPY[purpose];

  // The result view, after the button's POST
  const receipt = first(params.receipt);
  if (receipt !== undefined) {
    const name = verifyReceipt(receipt, purpose);
    return name ? (
      <Message role="status" text={copy.done(name)} />
    ) : (
      <Message role="alert" text={INVALID_LINK} />
    );
  }
  const error = first(params.error);
  if (error !== undefined) {
    return (
      <Message
        role="alert"
        text={error === "unavailable" ? UNAVAILABLE : INVALID_LINK}
      />
    );
  }

  // The landing view: read-only lookup, then one button
  const token = first(params.token);
  const lookup = await lookupNotifyToken(token, purpose);
  if (!lookup.ok) {
    return (
      <Message
        role="alert"
        text={lookup.reason === "unavailable" ? UNAVAILABLE : INVALID_LINK}
      />
    );
  }
  return (
    <Shell>
      <h1 className="text-lg font-semibold text-gray-100">
        {copy.heading(lookup.topicName)}
      </h1>
      <p className="text-sm text-gray-300">{copy.body}</p>
      <form method="post" action={`/api/notify/${purpose}`}>
        <input type="hidden" name="token" value={token} />
        <button
          type="submit"
          className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
        >
          {copy.button}
        </button>
      </form>
    </Shell>
  );
}
