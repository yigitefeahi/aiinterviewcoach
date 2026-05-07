/** Coerce API/UI values to readable string (avoid "[object Object]" in React or TTS). */
export function safeText(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value;
  if (value == null) return fallback;
  if (typeof value === "object") {
    const o = value as Record<string, unknown>;
    if (typeof o.detail === "string") return o.detail;
    if (typeof o.message === "string") return o.message;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

/** For speech: plain string only, never pass arbitrary objects to Utterance. */
export function speakableText(value: unknown): string {
  const s = safeText(value, "").trim();
  return s || "No feedback text available.";
}
