"""Lightweight section detection for résumé text (heuristic, multilingual-friendly)."""

from __future__ import annotations

import re
from typing import Any


_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("experience", re.compile(r"(?im)^(experience|work experience|employment|professional experience|iş deneyimi|deneyim)\s*[:.\-]?\s*$")),
    ("education", re.compile(r"(?im)^(education|academic|qualifications|eğitim)\s*[:.\-]?\s*$")),
    ("skills", re.compile(r"(?im)^(skills|technical skills|competencies|yetenekler)\s*[:.\-]?\s*$")),
    ("projects", re.compile(r"(?im)^(projects|selected projects|projeler)\s*[:.\-]?\s*$")),
]


def extract_cv_sections(text: str) -> dict[str, Any]:
    """Split text into coarse sections when headings are found; otherwise return summary stats."""
    raw = (text or "").strip()
    if not raw:
        return {"headings_found": [], "lines": 0, "chars": 0}

    lines = raw.splitlines()
    sections: dict[str, list[str]] = {}
    current = "summary"
    sections[current] = []

    for line in lines:
        stripped = line.strip()
        matched = False
        for key, pat in _SECTION_PATTERNS:
            if pat.match(stripped):
                current = key
                if key not in sections:
                    sections[key] = []
                matched = True
                break
        if not matched:
            sections.setdefault(current, []).append(line)

    headings_found = [k for k in sections if k != "summary" and sections[k]]
    return {
        "headings_found": headings_found,
        "lines": len(lines),
        "chars": len(raw),
        "experience_excerpt": "\n".join(sections.get("experience", [])[:40]).strip() or None,
        "education_excerpt": "\n".join(sections.get("education", [])[:25]).strip() or None,
        "skills_excerpt": "\n".join(sections.get("skills", [])[:25]).strip() or None,
    }
