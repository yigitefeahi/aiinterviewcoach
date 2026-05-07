export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://localhost:8000");

const CSRF_STORAGE_KEY = "aicoach_csrf";

/** Stateless CSRF token from login/register (HMAC-signed on the server). */
let csrfToken: string | null = null;

function readStoredCsrf(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return sessionStorage.getItem(CSRF_STORAGE_KEY);
  } catch {
    return null;
  }
}

function effectiveCsrf(): string | null {
  return csrfToken || readStoredCsrf();
}

export function setCsrfToken(token: string | null) {
  csrfToken = token;
  if (typeof window === "undefined") return;
  try {
    if (token) sessionStorage.setItem(CSRF_STORAGE_KEY, token);
    else sessionStorage.removeItem(CSRF_STORAGE_KEY);
  } catch {
    /* private mode / quota */
  }
}

export function getCsrfToken(): string | null {
  return effectiveCsrf();
}

export async function clearToken() {
  csrfToken = null;
  if (typeof window !== "undefined") {
    try {
      sessionStorage.removeItem(CSRF_STORAGE_KEY);
    } catch {
      /* */
    }
  }
  if (typeof window === "undefined") return;
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
  } catch {
    /* ignore network errors on logout */
  }
}

/** Headers for raw fetch() calls (e.g. TTS) that do not use apiFetch. */
export function buildAuthHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  const t = effectiveCsrf();
  if (t) h["X-CSRF-Token"] = t;
  return h;
}

export async function apiFetch(path: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers || {});

  const t = effectiveCsrf();
  if (t) {
    headers.set("X-CSRF-Token", t);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      credentials: "include",
      headers,
    });
  } catch {
    throw new Error("We could not reach the AI Coach server. Please check that the backend is running and try again.");
  }

  if (!res.ok) {
    let message =
      res.status === 429
        ? "You are moving a little too fast. Please wait a moment and try again."
        : res.status >= 500
          ? "Something went wrong on our side. Please try again in a moment."
          : `Request failed: ${res.status}`;
    try {
      const data = await res.json();
      message = data?.detail || JSON.stringify(data);
    } catch {
      const text = await res.text().catch(() => "");
      if (text) message = text;
    }
    throw new Error(message);
  }

  return res;
}

/** @deprecated Prefer checking session via GET /auth/me */
export function isAuthenticated() {
  return !!effectiveCsrf();
}
