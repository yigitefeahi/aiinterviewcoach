import { defineConfig, devices } from "@playwright/test";

const isCi = !!process.env.CI;
/** CI uses a free port; local dev defaults to 3000. */
const devPort = process.env.PLAYWRIGHT_DEV_PORT ?? (isCi ? "9323" : "3000");
const devHost = process.env.PLAYWRIGHT_HOST ?? "127.0.0.1";
const defaultBaseURL = `http://${devHost}:${devPort}`;
const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? defaultBaseURL;

/**
 * In CI we use `next build` + `next start` so we do not fight the single `.next/dev/lock`
 * that would block a second `next dev` in the same repo (e.g. while you have dev on :3000).
 */
const webServer = isCi
  ? {
      command: `npm run build && npm run start -- -H ${devHost} -p ${devPort}`,
      url: defaultBaseURL,
      reuseExistingServer: false,
      timeout: 300_000,
    }
  : {
      command: `npm run dev -- -H ${devHost} -p ${devPort}`,
      url: defaultBaseURL,
      reuseExistingServer: true,
      timeout: 120_000,
    };

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: isCi,
  retries: isCi ? 2 : 0,
  use: {
    baseURL,
    trace: "on-first-retry",
  },
  webServer,
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chrome", use: { ...devices["Pixel 5"] } },
  ],
});
