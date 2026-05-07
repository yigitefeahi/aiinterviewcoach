"""Optional LLM pass on top of keyword-based CV suggestions."""

import json
from typing import Any

from .config import settings
from .openai_client import client


def maybe_enrich_cv_suggestions(cv_text: str, base: dict[str, Any]) -> dict[str, Any]:
    if not settings.cv_llm_enrich:
        return base
    text = (cv_text or "").strip()
    if len(text) < 60:
        return base
    professions = base.get("suggested_professions") or []
    sectors = base.get("suggested_sectors") or []
    prompt = {
        "task": (
            "Given CV text and baseline keyword suggestions, return JSON with keys: "
            "suggested_professions (array of 1-3 strings), suggested_sectors (array of 1-3 strings), "
            "rationale (short string). Refine order only if clearly justified by CV content; "
            "do not invent employers or degrees."
        ),
        "baseline_professions": professions,
        "baseline_sectors": sectors,
        "cv_excerpt": text[:6000],
    }
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw else {}
        out = dict(base)
        if isinstance(data.get("suggested_professions"), list) and data["suggested_professions"]:
            out["suggested_professions"] = [str(x) for x in data["suggested_professions"][:3]]
        if isinstance(data.get("suggested_sectors"), list) and data["suggested_sectors"]:
            out["suggested_sectors"] = [str(x) for x in data["suggested_sectors"][:3]]
        if isinstance(data.get("rationale"), str) and data["rationale"].strip():
            out["rationale"] = data["rationale"].strip()
        if base.get("method") == "keyword_embedding_fusion":
            out["method"] = "keyword_embedding_fusion_plus_llm_refine"
        else:
            out["method"] = "keyword_heuristic_plus_llm_refine"
        out["limitations"] = (
            base.get("limitations", "")
            + " LLM refinement re-ranks baseline suggestions; it is not verified ground truth."
        ).strip()
        return out
    except Exception:
        return base
