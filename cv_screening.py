"""Single-pass LLM CV screening: role fit + short evaluator narrative (token-conscious)."""

import json
from typing import Any

from .config import settings
from .openai_client import client


def apply_cv_screening_llm(cv_text: str, baseline: dict[str, Any], intended_profession: str) -> dict[str, Any]:
    """
    Refines profession/sector picks and adds evaluator-style feedback.
    Falls back to baseline on any error.
    """
    text = (cv_text or "").strip()
    if len(text) < 80:
        return baseline

    excerpt = text[:12000]
    profs = baseline.get("suggested_professions") or []
    secs = baseline.get("suggested_sectors") or []
    cv_structure = baseline.get("cv_structure")
    struct_blob = ""
    if isinstance(cv_structure, dict) and cv_structure:
        try:
            struct_blob = json.dumps(cv_structure, ensure_ascii=False)[:6000]
        except Exception:
            struct_blob = ""

    screening_model = (settings.cv_screening_model or "").strip() or settings.llm_model

    prompt = {
        "instructions": (
            "You are an experienced hiring screener. Given CV text, optional parsed sections, and baseline keyword picks, "
            "output JSON only. "
            "Refine suggested_professions (1-3) and suggested_sectors (1-3) from the baseline lists when the CV clearly supports it; "
            "otherwise keep baseline order. "
            "Be careful not to over-rank Software Engineer just because a CV mentions Python, SQL, or technical tools; "
            "if the evidence is dashboards, KPIs, reporting, Excel, Power BI, Tableau, requirements, stakeholders, UAT, or process analysis, "
            "prefer Data Analyst or Business Analyst when those are in the baseline. "
            "intended_profession is the user's selected target role in the app — you MUST assess fit for that exact role, "
            "not only the top suggested role. "
            "Ground strengths and weaknesses in specific signals from the excerpt (skills, titles, impact metrics, tenure). "
            "Do not invent employers, degrees, certifications, or dates not supported by the text. "
            "If extraction looks noisy or the CV is very short, say so in weaknesses and lower confidence in fit. "
            "Evaluator copy must read like a human screener: headline is one punchy line; for_role_note is 2-4 sentences."
        ),
        "intended_profession": (intended_profession or "").strip() or "Unknown",
        "baseline_professions": profs,
        "baseline_sectors": secs,
        "role_fit_breakdown": baseline.get("role_fit_breakdown"),
        "cv_parsed_sections_json": struct_blob or None,
        "cv_excerpt": excerpt,
    }

    try:
        resp = client.chat.completions.create(
            model=screening_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You output only valid JSON with keys: "
                        "suggested_professions (array of strings), suggested_sectors (array of strings), "
                        "rationale (short string explaining the ranking vs baseline), "
                        "evaluator (object with: headline string, fit one of strong|moderate|weak|unclear, "
                        "strengths array max 4 short bullet strings, weaknesses array max 4 short bullet strings, "
                        "for_role_note string 2-4 sentences on fit specifically for intended_profession, "
                        "disclaimer string one sentence that this is AI screening not a hiring decision)."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw else {}
        out = dict(baseline)

        if isinstance(data.get("suggested_professions"), list) and data["suggested_professions"]:
            out["suggested_professions"] = [str(x) for x in data["suggested_professions"][:3]]
        if isinstance(data.get("suggested_sectors"), list) and data["suggested_sectors"]:
            out["suggested_sectors"] = [str(x) for x in data["suggested_sectors"][:3]]
        if isinstance(data.get("rationale"), str) and data["rationale"].strip():
            out["rationale"] = data["rationale"].strip()

        ev = data.get("evaluator")
        if isinstance(ev, dict):
            out["evaluator"] = {
                "headline": str(ev.get("headline", "")).strip() or "Screening summary unavailable.",
                "fit": str(ev.get("fit", "unclear")).strip().lower()
                if str(ev.get("fit", "")).strip()
                else "unclear",
                "strengths": [str(x) for x in (ev.get("strengths") or [])[:4] if str(x).strip()],
                "weaknesses": [str(x) for x in (ev.get("weaknesses") or [])[:4] if str(x).strip()],
                "for_role_note": str(ev.get("for_role_note", "")).strip(),
                "disclaimer": str(ev.get("disclaimer", "")).strip()
                or "AI-assisted screening; not a hiring decision.",
            }

        base_method = str(baseline.get("method") or "keyword_heuristic")
        if "embedding" in base_method:
            out["method"] = "keyword_embedding_fusion_plus_screening"
        else:
            out["method"] = "keyword_heuristic_plus_screening"

        prev_lim = baseline.get("limitations") or ""
        out["limitations"] = (
            f"{prev_lim} Screening uses one LLM pass ({screening_model}); verify facts on the real CV/PDF.".strip()
        )
        return out
    except Exception:
        return baseline
