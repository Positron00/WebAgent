import { test, expect } from "@playwright/test";
import {
  api,
  header,
  mails,
  signIn,
  waitForMail,
  waitForText,
} from "./helpers";

test("subscribe, confirm by button, get a notice, unsubscribe in one click", async ({
  page,
}) => {
  const name = `E2E email ${Date.now()}`;
  const address = `reader-${Date.now()}@e2e.test`;
  await signIn(page);

  // A topic straight from the API proxy (the research flow is covered by the other spec).
  const created = await api(page, "POST", "/api/topics", {
    name,
    query: "Latest results in protein folding research",
    schedule: {
      every: "week",
      weekday: "mon",
      time: "08:00",
      tz: "Europe/Paris",
    },
  });
  expect(created.status).toBe(201);
  const topicId = created.json.topic_id as string;

  await page.goto(`/topics/${topicId}`);
  const seen = mails().length;
  await page.getByLabel("Add an email address").fill(address);
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.getByText(/Waiting for confirmation/)).toBeVisible();

  const confirm = await waitForMail(
    (m) =>
      header(m, "To") === address && /notify\/confirm\?token=/.test(m.body),
    seen,
  );
  const link = confirm.body.match(/(http\S+\/notify\/confirm\?token=\S+)/)![1];

  // Opening the link changes nothing (mail scanners open links); the button confirms.
  await page.goto(link);
  await expect(page.getByRole("button", { name: "Confirm" })).toBeVisible();
  const before = (await api(page, "GET", `/api/topics/${topicId}`)).json;
  expect(before.notify.subscribers[0].confirmed_at).toBeFalsy();
  await page.getByRole("button", { name: "Confirm" }).click();
  await expect(
    page.getByText(/Confirmed: you'll get emails about/),
  ).toBeVisible();

  // A run sends a notice: plain text, no model text, one-click unsubscribe header kept literal.
  const seenBeforeRun = mails().length;
  await page.goto("/topics");
  await page.getByRole("button", { name: `Run ${name} now` }).click();
  const notice = await waitForMail(
    (m) => header(m, "To") === address && /Read the report/.test(m.body),
    seenBeforeRun,
    200_000,
  );
  expect(notice.body).not.toMatch(/fake completion/i);
  const listUnsubscribe = header(notice, "List-Unsubscribe");
  expect(listUnsubscribe).toMatch(/^<http[^>]+>$/);
  expect(header(notice, "List-Unsubscribe-Post")).toBe(
    "List-Unsubscribe=One-Click",
  );

  // A mail client sends this with no session, so the plain request client is the right test.
  const oneClick = await page.request.post(listUnsubscribe.slice(1, -1), {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    data: "List-Unsubscribe=One-Click",
  });
  expect(await oneClick.json()).toEqual({ status: "unsubscribed" });
  const after = (await api(page, "GET", `/api/topics/${topicId}`)).json;
  expect(after.notify.subscribers).toEqual([]);
  await page.goto(`/topics/${topicId}`);
  await waitForText(page, "First report");
});
