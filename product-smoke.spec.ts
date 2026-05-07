import { expect, type Page, test } from "@playwright/test";

async function mockApi(page: Page) {
  await page.route("**/auth/me", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ id: 1, name: "Demo User", email: "demo@example.com", profession: "Frontend Developer" }),
    });
  });
  await page.route("**/interview/sessions", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        sessions: [
          {
            session_id: 1,
            profession: "Frontend Developer",
            created_at: new Date().toISOString(),
            average_score: 84,
            completed: true,
            done: true,
            turn_count: 5,
            config: { mode: "text", focus_area: "Behavioral", difficulty: "Mid" },
          },
        ],
      }),
    });
  });
  await page.route("**/analytics/progress", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        summary: { sessions: 1, scored_sessions: 1, average_score: 84, best_score: 84, trend: 4 },
        timeline: [],
        focus_breakdown: [{ focus: "Behavioral", average_score: 84, turns: 5 }],
        next_best_action: "Run one focused drill on your lowest scoring area.",
      }),
    });
  });
  await page.route("**/quality/questions", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        summary: { questions_scanned: 5, unique_questions: 5, duplicate_count: 0, freshness_score: 100 },
        by_focus: { Behavioral: 5 },
        by_mode: { text: 5 },
        recent_questions: [],
        recommendation: "Question freshness is healthy.",
      }),
    });
  });
  await page.route("**/account/summary", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        user: { name: "Demo User", email: "demo@example.com", profession: "Frontend Developer" },
        usage: { sessions: 1, completed_sessions: 1, turns: 5, stories: 1, completed_drills: 2, average_score: 84 },
        preferences: {
          target_company: "Google",
          interview_date: "2026-05-20",
          default_mode: "text",
          focus_area: "Behavioral",
          difficulty: "Mid",
        },
        privacy: {
          cv_processing: "Uploaded CV files are processed for suggestions and not stored as files by this API.",
          interview_data: "Interview data is stored under your account.",
          delete_data_endpoint: "/account/data",
        },
      }),
    });
  });
  await page.route("**/account/usage-guards", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        status: "active",
        limits: [{ scope: "answer-text", max_requests: 20, window_seconds: 60, used_in_window: null }],
        cost_controls: ["Answer, pass, hint, audio, and video paths are rate limited."],
      }),
    });
  });
}

test("dashboard renders as product hub on desktop and mobile", async ({ page }) => {
  await mockApi(page);
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: /interview coaching hub/i })).toBeVisible();
  await expect(page.getByText(/progress analytics/i)).toBeVisible();
  await expect(page.getByText("Question Freshness", { exact: true })).toBeVisible();
});

test("settings exposes privacy, defaults, and usage guards", async ({ page }) => {
  await mockApi(page);
  await page.goto("/settings");
  await expect(page.getByRole("heading", { name: /profile, privacy, and defaults/i })).toBeVisible();
  await expect(page.getByText(/cost & usage guards/i)).toBeVisible();
  await expect(page.locator('input[value="Google"]')).toBeVisible();
});
