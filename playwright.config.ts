import { defineConfig, devices } from "@playwright/test";

// End-to-end tests against the stack from scripts/e2e/stack.sh (fake Together, SMTP sink, backend,
// scheduler and the built web app, all on 127.0.0.1). Start it first: `npm run e2e:stack`.
export default defineConfig({
  testDir: "./e2e",
  // A research run with the fake model takes about 15 s, and the scheduler ticks every 60 s.
  timeout: 240_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://127.0.0.1:3100",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
