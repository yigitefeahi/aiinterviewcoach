import { test, expect } from "@playwright/test";

test("landing page responds", async ({ page }) => {
  test.skip(
    !!process.env.CI,
    "Run locally with backend + `npm run dev`, then `npm run test:e2e`.",
  );
  await page.goto("/");
  await expect(page.locator("body")).toBeVisible();
});
