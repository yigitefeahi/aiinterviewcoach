from typing import Any, Optional
import json
import re
import statistics

from .config import settings
from .openai_client import client
from .rag import retrieve_for_evaluation, build_citations, evaluate_retrieval
from .product_features import (
    SCORECARD_DIMENSIONS,
    analyze_tone,
    build_company_prompt_context,
    expand_scorecard,
)

SYSTEM_PROMPT = """
You are a senior AI interview coach.

You are evaluating a mock interview answer for a specific profession and interview configuration.

Your task:
1. Score the answer from 0 to 100.
2. Score these rubric dimensions from 0 to 100:
   - technical_depth
   - communication
   - problem_solving
   - confidence
   - clarity
   - structure
   Also use company_context and case_type when provided.
3. Give clear strengths.
4. Give clear weaknesses.
5. Give concise actionable suggestions.
6. Provide recommended next steps.
7. Decide whether the interview is done.
8. Provide the next interview question if interview continues.

Rules:
- Use retrieved_context only if it is relevant.
- Use previous_turns to avoid repeating generic feedback patterns.
- The next question should fit the profession, difficulty and focus area.
- Be practical and interview-like.
- Keep feedback concise but useful.
- Feedback must reference at least 2 concrete details from the candidate's answer.
- If answer quality is weak, explain exactly what is missing.

Return strictly valid JSON with keys:
- score
- sub_scores
- strengths
- weaknesses
- suggestions
- recommended_next_steps
- feedback
- next_question
- done
"""

RUBRIC_KEYS = [
    "communication",
    "technical_depth",
    "confidence",
    "clarity",
    "structure",
    "problem_solving",
]


def safe_json_loads(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        return {}


def _build_fallback_feedback(answer_text: str, question: str) -> dict[str, Any]:
    wc = len(answer_text.split())
    has_numbers = bool(any(ch.isdigit() for ch in answer_text))
    has_structure = any(token in answer_text.lower() for token in ["first", "then", "finally", "because", "result"])
    technical = any(token in answer_text.lower() for token in ["api", "database", "latency", "debug", "test", "architecture"])
    score = 45 + (15 if wc > 60 else 0) + (10 if wc > 120 else 0) + (10 if has_structure else 0) + (10 if technical else 0) + (5 if has_numbers else 0)
    score = max(0, min(100, score))
    return {
        "score": score,
        "sub_scores": {
            "communication": score,
            "technical_depth": score if technical else max(35, score - 15),
            "confidence": max(35, score - 5),
            "clarity": score if has_structure else max(35, score - 10),
            "structure": score if has_structure else max(35, score - 12),
            "problem_solving": score if technical else max(35, score - 8),
        },
        "strengths": ["You addressed the question directly.", "Your response included relevant context from your own experience."],
        "weaknesses": ["The answer needs stronger measurable outcomes and clearer impact.", "Add more concrete technical detail tied to tools or decisions."],
        "suggestions": ["Use a concise STAR structure: Situation, Task, Action, Result.", "Add one metric to show impact (latency, conversion, cost, or delivery speed)."],
        "recommended_next_steps": ["Practice one 90-second answer using a measurable result.", "Prepare two project stories with trade-offs and outcomes."],
        "feedback": f"You answered the question but the response can be more interview-ready. For '{question}', improve structure and include one measurable outcome.",
        "next_question": None,
        "done": False,
    }


def _detect_red_flags(question: str, answer_text: str, score: int) -> list[str]:
    flags: list[str] = []
    q_tokens = set(re.findall(r"[a-zA-Z0-9_]+", (question or "").lower()))
    a_tokens = re.findall(r"[a-zA-Z0-9_]+", (answer_text or "").lower())
    answer_set = set(a_tokens)
    if len(a_tokens) < 25:
        flags.append("too_short")
    overlap_ratio = (len(q_tokens & answer_set) / max(1, len(q_tokens))) if q_tokens else 0.0
    if overlap_ratio < 0.12:
        flags.append("possibly_off_topic")
    if not any(ch.isdigit() for ch in answer_text):
        flags.append("no_measurable_metric")
    if sum(1 for term in {"thing", "stuff", "maybe", "kind of", "somehow", "etc"} if term in answer_text.lower()) >= 2:
        flags.append("vague_language")
    if score < 55 and "no_measurable_metric" not in flags:
        flags.append("low_signal_answer")
    return list(dict.fromkeys(flags))


def _compute_confidence(sub_scores: dict[str, int], retrieval_evidence: list[dict[str, Any]], answer_text: str, red_flags: list[str]) -> int:
    sub_avg = int(sum(sub_scores.values()) / max(1, len(sub_scores))) if sub_scores else 0
    top_hybrid = float(retrieval_evidence[0].get("hybrid_score", 0) or 0) if retrieval_evidence else 0.0
    richness = min(100, max(0, len(answer_text.split())))
    confidence = int((0.55 * sub_avg) + (0.30 * (top_hybrid * 100)) + (0.15 * richness))
    confidence -= min(25, len(red_flags) * 6)
    return max(0, min(100, confidence))


def _empty_retrieval_quality() -> dict[str, Any]:
    return {
        "label": "none",
        "score": 0,
        "top_score": 0,
        "source_count": 0,
        "doc_type_count": 0,
        "evidence_count": 0,
    }


def _build_score_explanation(score: int, sub_scores: dict[str, int], red_flags: list[str]) -> str:
    ranked = sorted(sub_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_text = ", ".join(name.replace("_", " ") for name, _ in ranked[:2]) if ranked else "overall response quality"
    low_text = ", ".join(name.replace("_", " ") for name, value in ranked if value < 60) or "none"
    base = f"Score {score}/100 is mainly driven by your strongest areas: {top_text}. Lower-scoring areas: {low_text}."
    if red_flags:
        return f"{base} Key risk signals detected: {', '.join(flag.replace('_', ' ') for flag in red_flags[:2])}."
    return base


def score_reliability(scores: list[int]) -> dict[str, Any]:
    if not scores:
        return {"runs": 0, "scores": [], "mean_score": 0, "min_score": 0, "max_score": 0, "std_dev": 0.0, "consistency_percent": 0, "consistency_label": "unknown"}
    mean_score = round(sum(scores) / len(scores), 2)
    std_dev = round(statistics.pstdev(scores), 2) if len(scores) > 1 else 0.0
    spread = max(scores) - min(scores)
    consistency_percent = max(0, min(100, int(round(100 - (std_dev * 8) - (spread * 1.2)))))
    label = "high" if consistency_percent >= 80 else "moderate" if consistency_percent >= 60 else "low"
    return {
        "runs": len(scores),
        "scores": scores,
        "mean_score": mean_score,
        "min_score": min(scores),
        "max_score": max(scores),
        "std_dev": std_dev,
        "consistency_percent": consistency_percent,
        "consistency_label": label,
    }


def evaluate_answer(
    profession: str,
    question: str,
    answer_text: str,
    config: Optional[dict[str, Any]] = None,
    previous_turns: Optional[list[dict[str, Any]]] = None,
    attempt_index: int = 1,
    use_rag: bool = True,
) -> dict[str, Any]:
    config = config or {}
    previous_turns = previous_turns or []
    context = ""
    retrieval_evidence: list[dict[str, Any]] = []
    rag_summary = "RAG disabled for this evaluation."
    retrieval_quality = _empty_retrieval_quality()
    if use_rag:
        rag_result = retrieve_for_evaluation(
            profession=profession,
            question=question,
            answer_text=answer_text,
            config=config,
        )
        context = rag_result.context
        retrieval_evidence = rag_result.evidence
        rag_summary = rag_result.summary
        retrieval_quality = rag_result.quality
    user_prompt = {
        "profession": profession,
        "difficulty": config.get("difficulty"),
        "mode": config.get("mode"),
        "interview_length": config.get("interview_length"),
        "focus_area": config.get("focus_area"),
        "sector": config.get("sector"),
        "target_company": config.get("target_company"),
        "company_context": build_company_prompt_context(config),
        "case_type": config.get("case_type"),
        "scorecard_dimensions": SCORECARD_DIMENSIONS,
        "question": question,
        "answer": answer_text,
        "attempt_index": attempt_index,
        "previous_turns": previous_turns[-5:],
        "retrieved_context": context,
        "retrieval_quality": retrieval_quality,
        "rag_summary": rag_summary,
        "instructions": (
            "Use retrieved_context only if helpful and relevant. "
            "When retrieval_quality is low, rely more on the answer and rubric than on weak sources. "
            "Be strict, specific, and evidence-based. Avoid generic repeated language."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}],
            response_format={"type": "json_object"},
            temperature=0.55,
        )
        data = safe_json_loads(resp.choices[0].message.content or "{}")
    except Exception:
        data = _build_fallback_feedback(answer_text, question)

    score = int(max(0, min(100, data.get("score", 0))))
    raw_sub_scores = data.get("sub_scores", {}) or {}
    sub_scores = {
        "communication": int(max(0, min(100, raw_sub_scores.get("communication", 0)))),
        "technical_depth": int(max(0, min(100, raw_sub_scores.get("technical_depth", raw_sub_scores.get("technical_accuracy", 0))))),
        "confidence": int(max(0, min(100, raw_sub_scores.get("confidence", 0)))),
        "clarity": int(max(0, min(100, raw_sub_scores.get("clarity", raw_sub_scores.get("communication", 0))))),
        "structure": int(max(0, min(100, raw_sub_scores.get("structure", raw_sub_scores.get("problem_solving", 0))))),
        "problem_solving": int(max(0, min(100, raw_sub_scores.get("problem_solving", 0)))),
    }
    strengths = [str(x).strip() for x in (data.get("strengths", []) if isinstance(data.get("strengths", []), list) else []) if str(x).strip()]
    weaknesses = [str(x).strip() for x in (data.get("weaknesses", []) if isinstance(data.get("weaknesses", []), list) else []) if str(x).strip()]
    suggestions = [str(x).strip() for x in (data.get("suggestions", []) if isinstance(data.get("suggestions", []), list) else []) if str(x).strip()]
    recommended_next_steps = [str(x).strip() for x in (data.get("recommended_next_steps", []) if isinstance(data.get("recommended_next_steps", []), list) else []) if str(x).strip()]
    feedback = str(data.get("feedback", "")).strip() or "Your answer was evaluated, but the model did not produce detailed feedback."
    next_q = str(data.get("next_question", "") or "").strip() or None
    done = bool(data.get("done", False))
    red_flags = _detect_red_flags(question, answer_text, score)
    confidence_score = _compute_confidence(sub_scores=sub_scores, retrieval_evidence=retrieval_evidence, answer_text=answer_text, red_flags=red_flags)
    scorecard = expand_scorecard(sub_scores, answer_text, config)
    tone_signals = analyze_tone(answer_text)
    company_rubric = build_company_prompt_context(config)["company_pack"]
    citations = build_citations(retrieval_evidence)
    rag_evaluation = evaluate_retrieval(retrieval_evidence, answer_text=answer_text, feedback_text=feedback)
    citation_notes = [
        f"{citation['id']}: {citation['doc_type']} evidence from {citation['source']} supports this feedback signal."
        for citation in citations[:3]
    ]
    return {
        "score": score,
        "sub_scores": sub_scores,
        "scorecard": scorecard,
        "tone_signals": tone_signals,
        "company_rubric": company_rubric,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "recommended_next_steps": recommended_next_steps,
        "feedback": feedback,
        "next_question": next_q,
        "done": done,
        "retrieval_evidence": retrieval_evidence,
        "rag_summary": rag_summary,
        "retrieval_quality": retrieval_quality,
        "citations": citations,
        "citation_notes": citation_notes,
        "rag_evaluation": rag_evaluation,
        "red_flags": red_flags,
        "confidence_score": confidence_score,
        "score_explanation": _build_score_explanation(score, sub_scores, red_flags),
    }
