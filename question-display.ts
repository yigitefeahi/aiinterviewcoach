/**
 * Same shape as backend `build_question_context`: role · level · focus · optional sector/company.
 */
export function buildSessionContextLine(params: {
  profession: string;
  difficulty: string;
  focusArea: string;
  sector?: string;
  company?: string;
}): string {
  const parts = [params.profession, params.difficulty, params.focusArea].filter(
    (x) => (x || "").trim().length > 0
  );
  const s = (params.sector || "").trim();
  const c = (params.company || "").trim();
  if (s) parts.push(`Sector: ${s}`);
  if (c) parts.push(`Target company: ${c}`);
  return parts.join(" · ");
}

/**
 * Older sessions stored one blob: preamble + "First question (...): ...".
 * Split for display when questionContext is not passed separately.
 */
export function splitLegacyQuestionBlock(full: string): { context?: string; body: string } {
  const t = (full || "").trim();
  if (!t) return { body: "" };
  const lower = t.toLowerCase();

  const sep = t.match(/\n-{3,}\n|\n\*{3,}\n/);
  if (sep && sep.index !== undefined) {
    const a = t.slice(0, sep.index).trim();
    const b = t.slice(sep.index + sep[0].length).trim();
    if (a.length >= 20 && b.length >= 10) {
      return { context: a, body: b };
    }
  }

  const qLine = t.match(/\n\nQuestion\s*[:\-]?\s*\n+/i);
  if (qLine && qLine.index !== undefined && qLine.index > 24) {
    const preamble = t.slice(0, qLine.index).trim();
    const rest = t.slice(qLine.index + qLine[0].length).trim();
    if (rest) return { context: preamble, body: rest };
  }

  const idx = lower.indexOf("first question");
  if (idx === -1) return { body: t };
  const preamble = t.slice(0, idx).replace(/\s+/g, " ").trim();
  let rest = t.slice(idx);
  rest = rest.replace(/^first question\s*\([^)]*\)\s*:\s*/i, "").trim();
  if (!rest) return { body: t };
  return { context: preamble || undefined, body: rest };
}
