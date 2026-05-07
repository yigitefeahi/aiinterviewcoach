"""Embedding-based profession affinity (OpenAI embeddings API)."""

from __future__ import annotations

import math
from .config import settings
from .openai_client import client


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def profession_embedding_scores(cv_text: str, professions: list[str]) -> dict[str, float]:
    """Return cosine similarity per profession label (0..1 scale, not calibrated to hiring)."""
    text = (cv_text or "").strip()
    if not text or not professions:
        return {}

    cv_excerpt = text[:8000]
    labels = [f"Job title / role: {p}" for p in professions]
    try:
        emb = client.embeddings.create(
            model=settings.embedding_model,
            input=[cv_excerpt] + labels,
        )
        vecs = [d.embedding for d in emb.data]
        if len(vecs) != len(professions) + 1:
            return {}
        cv_vec = vecs[0]
        out: dict[str, float] = {}
        for i, p in enumerate(professions):
            sim = _cosine(cv_vec, vecs[i + 1])
            # Map cosine [-1,1] to [0,1] for easier fusion with keyword counts
            out[p] = max(0.0, min(1.0, (sim + 1.0) / 2.0))
        return out
    except Exception:
        return {}


def fuse_keyword_and_embedding(
    keyword_ranked: list[tuple[str, int]],
    emb_scores: dict[str, float],
    *,
    keyword_weight: float = 0.55,
) -> list[tuple[str, float]]:
    """Combine keyword counts with embedding similarity into a single score."""
    if not keyword_ranked:
        return []
    max_kw = max((s for _, s in keyword_ranked), default=1) or 1
    fused: list[tuple[str, float]] = []
    for prof, kw in keyword_ranked:
        kw_n = kw / max_kw
        emb = emb_scores.get(prof, 0.0)
        score = keyword_weight * kw_n + (1.0 - keyword_weight) * emb
        fused.append((prof, score))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
