import { test, expect } from "@playwright/test";
import { signIn, waitForText } from "./helpers";

test("sign-in gates the app", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/login/);
  await page.getByLabel("Access code").fill("not-the-code-1234");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/login/);
  const api = await page.request.get("/api/topics");
  expect(api.status()).toBe(401);
  await signIn(page);
});

test("a research answer becomes a topic that the scheduler runs", async ({
  page,
}) => {
  const name = `E2E topic ${Date.now()}`;
  await signIn(page);

  // Agentic is Always for new users here (WEBAGENT_DEFAULT_AGENTIC_MODE=always): a research run.
  await page
    .getByPlaceholder("Ask anything...")
    .fill("What changed in open-weight LLM benchmarks this month?");
  await page.getByRole("button", { name: "Submit message" }).click();
  await expect(page.getByText("Deep research").first()).toBeVisible({
    timeout: 180_000,
  });

  await page
    .getByRole("button", { name: "Track this question on a schedule" })
    .click();
  await page.getByLabel("Name").fill(name);
  await page.getByRole("button", { name: "Create topic" }).click();
  await expect(page.getByText("Topic created")).toBeVisible();

  await page.goto("/topics");
  await expect(page.getByRole("link", { name })).toBeVisible();
  await page.getByRole("button", { name: `Run ${name} now` }).click();

  await page.getByRole("link", { name }).click();
  await waitForText(page, "First report");
  await page
    .getByRole("button", { name: /Open report/ })
    .first()
    .click();
  await expect(page.getByLabel("Stored report")).toBeVisible();
});
