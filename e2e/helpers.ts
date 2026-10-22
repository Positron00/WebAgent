import { readFileSync, readdirSync, existsSync } from "node:fs";
import path from "node:path";
import { expect, type Page } from "@playwright/test";

const E2E_DIR = process.env.E2E_DIR || path.join(__dirname, "..", ".e2e");

/** The throwaway secrets scripts/e2e/stack.sh wrote (never real ones). */
export function e2eEnv(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of readFileSync(path.join(E2E_DIR, "env"), "utf8").split(
    "\n",
  )) {
    const at = line.indexOf("=");
    if (at > 0) out[line.slice(0, at)] = line.slice(at + 1).trim();
  }
  return out;
}

export async function signIn(page: Page): Promise<void> {
  await page.goto("/login");
  await page.getByLabel("Access code").fill(e2eEnv().E2E_ACCESS_CODE);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByPlaceholder("Ask anything...")).toBeVisible();
}

export type Mail = {
  headers: Record<string, string>;
  body: string;
  file: string;
};

/** Messages the SMTP sink saved, oldest first. */
export function mails(): Mail[] {
  const dir = path.join(E2E_DIR, "mail");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .sort((a, b) => parseInt(a, 10) - parseInt(b, 10))
    .map((f) => ({
      ...JSON.parse(readFileSync(path.join(dir, f), "utf8")),
      file: f,
    }));
}

/** Wait for a new message (beyond `seen`) that matches `test`. */
export async function waitForMail(
  test: (m: Mail) => boolean,
  seen: number,
  timeoutMs = 120_000,
): Promise<Mail> {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    const found = mails().slice(seen).find(test);
    if (found) return found;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("no matching email arrived");
}

export function header(mail: Mail, name: string): string {
  const key = Object.keys(mail.headers).find(
    (k) => k.toLowerCase() === name.toLowerCase(),
  );
  return key ? mail.headers[key] : "";
}

/** Reload `page` until `text` is visible (the scheduler ticks every 60 s). */
export async function waitForText(
  page: Page,
  text: string | RegExp,
  timeoutMs = 200_000,
): Promise<void> {
  const until = Date.now() + timeoutMs;
  while (Date.now() < until) {
    // Give the page a moment to load its data after each reload; a bare isVisible() checks too early.
    const shown = await page
      .getByText(text)
      .first()
      .waitFor({ state: "visible", timeout: 5000 })
      .then(
        () => true,
        () => false,
      );
    if (shown) return;
    await page.reload();
  }
  await expect(page.getByText(text).first()).toBeVisible({ timeout: 1000 });
}

/** A call to the app's own API from inside the page, so it carries the session cookie. The cookie is
 *  Secure, and Playwright's request client won't send a Secure cookie over plain http. */
export async function api(
  page: Page,
  method: string,
  url: string,
  body?: unknown,
) {
  return page.evaluate(
    async ({ method, url, body }) => {
      const res = await fetch(url, {
        method,
        headers:
          body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      return { status: res.status, json: await res.json().catch(() => null) };
    },
    { method, url, body },
  );
}
