from typing import Any, Optional
import json

from sqlalchemy.orm import Session

from .models import InterviewSession, InterviewTurn, User, UserMemoryItem
from .openai_client import client
from .config import settings
from .interview_config import (
    normalize_config,
    target_question_count,
    generate_first_question,
    extract_topic_hint,
    generate_dynamic_question,
    dedupe_question,
    build_question_context,
    generate_bank_question,
)
from .interview_evaluation import (
    RUBRIC_KEYS,
    evaluate_answer,
    score_reliability,
)
from .rag import retrieve_for_question_generation

MAX_ATTEMPTS_PER_QUESTION = 2
MAX_PASSES_PER_SESSION = 3
RECENT_SESSION_QUESTION_LIMIT = 80
MEMORY_LIMIT = 18


def _session_question_context(session: InterviewSession) -> str:
    cfg = normalize_config((session.result_json or {}).get("config", {}))
    return build_question_context(session.profession, cfg)


def _question_retrieval(profession: str, config: dict[str, Any], asked_topics: list[str], avoid_questions: list[str]) -> dict[str, Any]:
    try:
        result = retrieve_for_question_generation(
            profession=profession,
            config=config,
            asked_topics=asked_topics,
            asked_questions=avoid_questions,
        )
        return {
            "context": result.context,
            "summary": result.summary,
            "quality": result.quality,
            "evidence": result.evidence,
        }
    except Exception:
        return {
            "context": "",
            "summary": "RAG question evidence unavailable; question bank fallback used.",
            "quality": {"label": "none", "score": 0, "evidence_count": 0},
            "evidence": [],
        }


def load_user_memory(db: Session, user: User, limit: int = MEMORY_LIMIT) -> list[dict[str, Any]]:
    rows = (
        db.query(UserMemoryItem)
        .filter(UserMemoryItem.user_id == user.id)
        .order_by(UserMemoryItem.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": row.id,
            "memory_type": row.memory_type,
            "content": row.content,
            "score": row.score,
            "meta": row.meta or {},
            "created_at": str(row.created_at),
        }
        for row in rows
    ]


def enrich_config_with_memory(db: Session, user: User, config: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(config)
    memories = load_user_memory(db, user)
    enriched["user_memory"] = memories
    enriched["cv_facts"] = [
        item["content"]
        for item in memories
        if item.get("memory_type") in {"cv_signal", "skill_gap", "role_strength"}
    ][:8]
    return enriched


def _memory_content_from_evaluation(question: str, answer_text: str, evaluation: dict[str, Any]) -> list[tuple[str, str, float, dict[str, Any]]]:
    scorecard = evaluation.get("scorecard") or {}
    low_dims = [name for name, value in sorted(scorecard.items(), key=lambda item: item[1]) if int(value or 0) < 65][:3]
    high_dims = [name for name, value in sorted(scorecard.items(), key=lambda item: item[1], reverse=True) if int(value or 0) >= 78][:2]
    memories: list[tuple[str, str, float, dict[str, Any]]] = []
    if low_dims:
        memories.append(
            (
                "weakness_pattern",
                f"User often needs practice in {', '.join(dim.replace('_', ' ') for dim in low_dims)}. Latest question: {question}",
                0.72,
                {"dimensions": low_dims, "score": evaluation.get("score")},
            )
        )
    if high_dims:
        memories.append(
            (
                "strength_pattern",
                f"User shows strength in {', '.join(dim.replace('_', ' ') for dim in high_dims)}. Reuse this strength in harder questions.",
                0.62,
                {"dimensions": high_dims, "score": evaluation.get("score")},
            )
        )
    red_flags = evaluation.get("red_flags") or []
    if red_flags:
        memories.append(
            (
                "risk_signal",
                f"User answer risk signals: {', '.join(str(flag).replace('_', ' ') for flag in red_flags[:3])}.",
                0.68,
                {"red_flags": red_flags[:5]},
            )
        )
    if any(ch.isdigit() for ch in answer_text):
        memories.append(("role_strength", "User has used measurable metrics in at least one answer.", 0.58, {"question": question}))
    else:
        memories.append(("skill_gap", "User should add measurable metrics or concrete impact to future answers.", 0.7, {"question": question}))
    return memories[:4]


def store_user_memory(db: Session, user: User, session: InterviewSession, question: str, answer_text: str, evaluation: dict[str, Any]) -> None:
    for memory_type, content, score, meta in _memory_content_from_evaluation(question, answer_text, evaluation):
        db.add(
            UserMemoryItem(
                user_id=user.id,
                session_id=session.id,
                memory_type=memory_type,
                content=content,
                score=score,
                meta=meta,
            )
        )


def _questions_from_result_json(result_json: dict[str, Any]) -> list[str]:
    questions: list[str] = []
    current = str(result_json.get("current_question") or "").strip()
    if current:
        questions.append(current)
    for turn in result_json.get("turns") or []:
        if isinstance(turn, dict):
            question = str(turn.get("question") or "").strip()
            if question:
                questions.append(question)
    return questions


def recent_user_questions(db: Session, user: User, exclude_session_id: Optional[int] = None) -> list[str]:
    query = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user.id)
        .order_by(InterviewSession.id.desc())
        .limit(12)
    )
    questions: list[str] = []
    for session in query.all():
        if exclude_session_id is not None and session.id == exclude_session_id:
            continue
        questions.extend(_questions_from_result_json(session.result_json or {}))
        if len(questions) >= RECENT_SESSION_QUESTION_LIMIT:
            break
    return questions[:RECENT_SESSION_QUESTION_LIMIT]


def serialize_answered_turns(turns: list[InterviewTurn]) -> list[dict[str, Any]]:
    return [
        {
            "id": t.id,
            "question": t.question,
            "answer": t.answer_text,
            "score": t.score,
            "feedback": t.ai_feedback,
            "created_at": str(t.created_at),
        }
        for t in turns
        if t.answer_text is not None
    ]


def build_in_progress_result_json(
    session: InterviewSession,
    config: dict[str, Any],
    current_question: Optional[str],
) -> dict[str, Any]:
    all_turns = sorted(session.turns, key=lambda t: t.id)
    answered_turns = [t for t in all_turns if t.answer_text is not None]
    scores = [t.score for t in answered_turns if t.score is not None]
    avg = int(sum(scores) / len(scores)) if scores else None

    return {
        "status": "in_progress",
        "config": config,
        "average_score": avg,
        "final_summary": None,
        "current_question": current_question,
        "turns": serialize_answered_turns(answered_turns),
    }


def build_completed_result_json(
    session: InterviewSession,
    config: dict[str, Any],
    last_evaluation: dict[str, Any],
    reference_answers: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    all_turns = sorted(session.turns, key=lambda t: t.id)
    answered_turns = [t for t in all_turns if t.answer_text is not None]
    scores = [t.score for t in answered_turns if t.score is not None]
    avg = int(sum(scores) / len(scores)) if scores else 0

    return {
        "status": "completed",
        "config": config,
        "average_score": avg,
        "final_summary": {
            "score": last_evaluation["score"],
            "sub_scores": last_evaluation["sub_scores"],
            "scorecard": last_evaluation.get("scorecard", {}),
            "tone_signals": last_evaluation.get("tone_signals", {}),
            "company_rubric": last_evaluation.get("company_rubric", {}),
            "score_explanation": last_evaluation.get("score_explanation"),
            "confidence_score": last_evaluation.get("confidence_score"),
            "red_flags": last_evaluation.get("red_flags", []),
            "strengths": last_evaluation["strengths"],
            "weaknesses": last_evaluation["weaknesses"],
            "suggestions": last_evaluation["suggestions"],
            "recommended_next_steps": last_evaluation["recommended_next_steps"],
            "feedback": last_evaluation["feedback"],
            "retrieval_evidence": last_evaluation.get("retrieval_evidence", []),
            "rag_summary": last_evaluation.get("rag_summary"),
            "retrieval_quality": last_evaluation.get("retrieval_quality"),
            "citations": last_evaluation.get("citations", []),
            "citation_notes": last_evaluation.get("citation_notes", []),
            "rag_evaluation": last_evaluation.get("rag_evaluation", {}),
            "reference_answers": reference_answers or [],
        },
        "current_question": None,
        "turns": serialize_answered_turns(answered_turns),
    }


def _build_reference_answers(
    profession: str,
    config: dict[str, Any],
    turns: list[InterviewTurn],
) -> list[dict[str, str]]:
    answered = [t for t in turns if t.answer_text]
    items = [{"question": t.question, "answer": t.answer_text} for t in answered][:8]
    if not items:
        return []
    prompt = {
        "profession": profession,
        "difficulty": config.get("difficulty"),
        "focus_area": config.get("focus_area"),
        "turns": items,
        "instructions": (
            "For each question, provide one concise high-quality reference answer (4-6 lines). "
            "Return JSON with key `reference_answers` containing [{question, sample_answer}]."
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You are a senior interview prep coach."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.45,
        )
        parsed = safe_json_loads(resp.choices[0].message.content or "{}")
        out = parsed.get("reference_answers", [])
        if not isinstance(out, list):
            return []
        normalized = []
        for it in out[:8]:
            if not isinstance(it, dict):
                continue
            q = str(it.get("question", "")).strip()
            a = str(it.get("sample_answer", "")).strip()
            if q and a:
                normalized.append({"question": q, "sample_answer": a})
        return normalized
    except Exception:
        return []


def create_session(
    db: Session,
    user: User,
    profession: str,
    config: Optional[dict[str, Any]] = None,
) -> tuple[InterviewSession, str, str]:
    config = normalize_config(config)
    avoid_questions = recent_user_questions(db, user)
    first_question_rag = _question_retrieval(profession, config, [], avoid_questions)
    question_context, first_q = generate_first_question(profession, config, avoid_questions=avoid_questions)

    session = InterviewSession(
        user_id=user.id,
        profession=profession,
        result_json={
            "status": "in_progress",
            "config": config,
            "average_score": None,
            "final_summary": None,
            "question_context": question_context,
            "question_rag": first_question_rag,
            "current_question": first_q,
            "turns": [],
            "asked_topics": [extract_topic_hint(first_q)],
            "recent_avoid_questions": avoid_questions[-30:],
            "passes_used": 0,
            "attempts_by_turn": {},
            "attempt_logs": [],
        },
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    turn = InterviewTurn(
        session_id=session.id,
        question=first_q,
        answer_text=None,
    )
    db.add(turn)
    db.commit()

    return session, first_q, question_context


def submit_answer(
    db: Session,
    user: User,
    session_id: int,
    answer_text: str,
) -> dict[str, Any]:
    session = db.get(InterviewSession, session_id)
    if not session or session.user_id != user.id:
        raise ValueError("Session not found")

    turns = sorted(session.turns, key=lambda t: t.id)
    if not turns:
        raise ValueError("No turns found")

    current_turn = turns[-1]
    if current_turn.answer_text is not None:
        raise ValueError("This question has already been answered")

    clean_answer = (answer_text or "").strip()
    if not clean_answer:
        raise ValueError("Answer cannot be empty")

    config = normalize_config((session.result_json or {}).get("config", {}))
    session_json = session.result_json or {}
    passes_used = int(session_json.get("passes_used", 0))
    passes_left = max(0, MAX_PASSES_PER_SESSION - passes_used)
    attempts_by_turn = dict(session_json.get("attempts_by_turn", {}))
    attempt_logs = list(session_json.get("attempt_logs", []))
    current_attempt = int(attempts_by_turn.get(str(current_turn.id), 0)) + 1
    attempts_left = max(0, MAX_ATTEMPTS_PER_QUESTION - current_attempt)
    previous_turns = serialize_answered_turns([t for t in turns if t.answer_text is not None])
    max_turns = target_question_count(config["interview_length"])
    current_question_index = len(previous_turns) + 1

    eval_config = enrich_config_with_memory(db, user, config)
    evaluation = evaluate_answer(
        session.profession,
        current_turn.question,
        clean_answer,
        config=eval_config,
        previous_turns=previous_turns,
        attempt_index=current_attempt,
    )

    can_retry = (
        attempts_left > 0
        and not evaluation.get("done", False)
        and int(evaluation.get("score", 0)) < 85
        and len(clean_answer) >= 20
    )

    if can_retry:
        attempts_by_turn[str(current_turn.id)] = current_attempt
        attempt_logs.append(
            {
                "turn_id": current_turn.id,
                "attempt": current_attempt,
                "question": current_turn.question,
                "answer": clean_answer,
                "score": evaluation["score"],
                "feedback": evaluation["feedback"],
            }
        )
        session_json["attempts_by_turn"] = attempts_by_turn
        session_json["attempt_logs"] = attempt_logs[-20:]
        session.result_json = build_in_progress_result_json(session, config, current_turn.question)
        session.result_json["attempts_by_turn"] = attempts_by_turn
        session.result_json["attempt_logs"] = attempt_logs[-20:]
        db.add(session)
        db.commit()

        return {
            "next_question": current_turn.question,
            "pending_next_question": evaluation.get("next_question"),
            "feedback": evaluation["feedback"],
            "score": evaluation["score"],
            "done": False,
            "can_retry": True,
            "attempts_left": attempts_left,
            "confidence_score": evaluation.get("confidence_score"),
            "red_flags": evaluation.get("red_flags", []),
            "score_explanation": evaluation.get("score_explanation"),
            "passes_left": passes_left,
            "question_index": current_question_index,
            "total_questions": max_turns,
            "sub_scores": evaluation["sub_scores"],
            "scorecard": evaluation.get("scorecard", {}),
            "tone_signals": evaluation.get("tone_signals", {}),
            "company_rubric": evaluation.get("company_rubric", {}),
            "strengths": evaluation["strengths"],
            "weaknesses": evaluation["weaknesses"],
            "suggestions": evaluation["suggestions"],
            "recommended_next_steps": evaluation["recommended_next_steps"],
            "retrieval_evidence": evaluation.get("retrieval_evidence", []),
            "rag_summary": evaluation.get("rag_summary"),
            "retrieval_quality": evaluation.get("retrieval_quality"),
            "citations": evaluation.get("citations", []),
            "citation_notes": evaluation.get("citation_notes", []),
            "rag_evaluation": evaluation.get("rag_evaluation", {}),
            "question_context": _session_question_context(session),
        }

    current_turn.answer_text = clean_answer
    current_turn.ai_feedback = evaluation["feedback"]
    current_turn.score = evaluation["score"]
    db.add(current_turn)
    db.flush()
    store_user_memory(db, user, session, current_turn.question, clean_answer, evaluation)

    answered_turns = [t for t in sorted(session.turns, key=lambda t: t.id) if t.answer_text is not None]
    max_turns = target_question_count(config["interview_length"])
    current_question_index = len(answered_turns) + 1

    next_question = evaluation["next_question"]
    done = evaluation["done"]

    if len(answered_turns) >= max_turns:
        done = True
        next_question = None

    asked_topics = list((session_json or {}).get("asked_topics", []))
    asked_questions = [t.question for t in sorted(session.turns, key=lambda t: t.id) if t.question]
    recent_questions = recent_user_questions(db, user, exclude_session_id=session.id)
    avoid_questions = (recent_questions + asked_questions)[-RECENT_SESSION_QUESTION_LIMIT:]
    fallback_question = generate_bank_question(session.profession, config, avoid_questions=avoid_questions)
    question_config = enrich_config_with_memory(db, user, config)
    question_rag = _question_retrieval(session.profession, question_config, asked_topics, avoid_questions)
    if next_question is None and not done:
        next_question = (
            generate_dynamic_question(
                session.profession,
                question_config,
                asked_topics,
                avoid_questions,
                retrieval_context=question_rag["context"],
                rag_summary=question_rag["summary"],
            )
            or fallback_question
        )
    if not done:
        next_question = dedupe_question(next_question, avoid_questions, fallback_question)

    if done:
        reference_answers = _build_reference_answers(session.profession, config, sorted(session.turns, key=lambda t: t.id))
        session.result_json = build_completed_result_json(
            session,
            config,
            evaluation,
            reference_answers=reference_answers,
        )
        db.add(session)
        db.commit()

        return {
            "next_question": None,
            "pending_next_question": None,
            "feedback": evaluation["feedback"],
            "score": evaluation["score"],
            "done": True,
            "can_retry": False,
            "attempts_left": 0,
            "confidence_score": evaluation.get("confidence_score"),
            "red_flags": evaluation.get("red_flags", []),
            "score_explanation": evaluation.get("score_explanation"),
            "passes_left": passes_left,
            "question_index": len(answered_turns),
            "total_questions": max_turns,
            "sub_scores": evaluation["sub_scores"],
            "scorecard": evaluation.get("scorecard", {}),
            "tone_signals": evaluation.get("tone_signals", {}),
            "company_rubric": evaluation.get("company_rubric", {}),
            "strengths": evaluation["strengths"],
            "weaknesses": evaluation["weaknesses"],
            "suggestions": evaluation["suggestions"],
            "recommended_next_steps": evaluation["recommended_next_steps"],
            "retrieval_evidence": evaluation.get("retrieval_evidence", []),
            "rag_summary": evaluation.get("rag_summary"),
            "retrieval_quality": evaluation.get("retrieval_quality"),
            "citations": evaluation.get("citations", []),
            "citation_notes": evaluation.get("citation_notes", []),
            "rag_evaluation": evaluation.get("rag_evaluation", {}),
            "question_context": _session_question_context(session),
        }

    new_turn = InterviewTurn(
        session_id=session.id,
        question=next_question,
        answer_text=None,
    )
    db.add(new_turn)
    db.flush()

    session.result_json = build_in_progress_result_json(session, config, next_question)
    session.result_json["asked_topics"] = (asked_topics + [extract_topic_hint(next_question)])[-20:]
    session.result_json["recent_avoid_questions"] = avoid_questions[-30:]
    session.result_json["passes_used"] = int((session_json or {}).get("passes_used", 0))
    session.result_json["attempts_by_turn"] = attempts_by_turn
    session.result_json["attempt_logs"] = attempt_logs[-20:]
    session.result_json["question_rag"] = question_rag
    db.add(session)
    db.commit()

    return {
        "next_question": next_question,
        "pending_next_question": None,
        "feedback": evaluation["feedback"],
        "score": evaluation["score"],
        "done": False,
        "can_retry": False,
        "attempts_left": 0,
        "confidence_score": evaluation.get("confidence_score"),
        "red_flags": evaluation.get("red_flags", []),
        "score_explanation": evaluation.get("score_explanation"),
        "passes_left": passes_left,
        "question_index": len(answered_turns) + 1,
        "total_questions": max_turns,
        "sub_scores": evaluation["sub_scores"],
        "scorecard": evaluation.get("scorecard", {}),
        "tone_signals": evaluation.get("tone_signals", {}),
        "company_rubric": evaluation.get("company_rubric", {}),
        "strengths": evaluation["strengths"],
        "weaknesses": evaluation["weaknesses"],
        "suggestions": evaluation["suggestions"],
        "recommended_next_steps": evaluation["recommended_next_steps"],
        "retrieval_evidence": evaluation.get("retrieval_evidence", []),
        "rag_summary": evaluation.get("rag_summary"),
        "retrieval_quality": evaluation.get("retrieval_quality"),
        "citations": evaluation.get("citations", []),
        "citation_notes": evaluation.get("citation_notes", []),
        "rag_evaluation": evaluation.get("rag_evaluation", {}),
        "question_context": _session_question_context(session),
    }


def pass_current_question(
    db: Session,
    user: User,
    session_id: int,
) -> dict[str, Any]:
    session = db.get(InterviewSession, session_id)
    if not session or session.user_id != user.id:
        raise ValueError("Session not found")
    turns = sorted(session.turns, key=lambda t: t.id)
    if not turns:
        raise ValueError("No turns found")
    current_turn = turns[-1]
    if current_turn.answer_text is not None:
        raise ValueError("This question has already been answered")

    session_json = session.result_json or {}
    passes_used = int(session_json.get("passes_used", 0))
    if passes_used >= MAX_PASSES_PER_SESSION:
        raise ValueError("No pass rights left for this session")

    current_turn.answer_text = "[PASSED]"
    current_turn.ai_feedback = "Question skipped by user."
    current_turn.score = 0
    db.add(current_turn)
    db.flush()

    config = normalize_config((session.result_json or {}).get("config", {}))
    answered_turns = [t for t in sorted(session.turns, key=lambda t: t.id) if t.answer_text is not None]
    max_turns = target_question_count(config["interview_length"])
    done = len(answered_turns) >= max_turns
    if done:
        dummy_eval = {
            "score": 0,
            "sub_scores": {k: 0 for k in RUBRIC_KEYS},
            "strengths": [],
            "weaknesses": ["Multiple questions were skipped. Complete more answers for a reliable evaluation."],
            "suggestions": ["Answer more questions to get meaningful coaching."],
            "recommended_next_steps": ["Run another session and avoid pass usage where possible."],
            "feedback": "Session ended after pass usage and question limit.",
            "retrieval_evidence": [],
            "confidence_score": 30,
            "red_flags": ["insufficient_answer_coverage"],
            "score_explanation": "Score is low because multiple questions were skipped, so signal quality is limited.",
        }
        reference_answers = _build_reference_answers(session.profession, config, sorted(session.turns, key=lambda t: t.id))
        session.result_json = build_completed_result_json(
            session,
            config,
            dummy_eval,
            reference_answers=reference_answers,
        )
        db.add(session)
        db.commit()
        return {
            "next_question": None,
            "pending_next_question": None,
            "feedback": "Pass applied. Session completed by question limit.",
            "score": 0,
            "done": True,
            "can_retry": False,
            "attempts_left": 0,
            "confidence_score": 30,
            "red_flags": ["insufficient_answer_coverage"],
            "passes_left": 0,
            "question_index": len(answered_turns),
            "total_questions": max_turns,
            "sub_scores": {k: 0 for k in RUBRIC_KEYS},
            "scorecard": {},
            "tone_signals": {},
            "company_rubric": {},
            "strengths": [],
            "weaknesses": ["Session contains skipped questions."],
            "suggestions": ["Run another session and answer all questions for better feedback."],
            "recommended_next_steps": ["Retry with full answers."],
            "retrieval_evidence": [],
            "question_context": _session_question_context(session),
        }

    asked_topics = list(session_json.get("asked_topics", []))
    asked_questions = [t.question for t in sorted(session.turns, key=lambda t: t.id) if t.question]
    recent_questions = recent_user_questions(db, user, exclude_session_id=session.id)
    avoid_questions = (recent_questions + asked_questions)[-RECENT_SESSION_QUESTION_LIMIT:]
    fallback_question = generate_bank_question(session.profession, config, avoid_questions=avoid_questions)
    question_config = enrich_config_with_memory(db, user, config)
    question_rag = _question_retrieval(session.profession, question_config, asked_topics, avoid_questions)
    next_question = (
        generate_dynamic_question(
            session.profession,
            question_config,
            asked_topics,
            avoid_questions,
            retrieval_context=question_rag["context"],
            rag_summary=question_rag["summary"],
        )
        or fallback_question
    )
    next_question = dedupe_question(next_question, avoid_questions, fallback_question)

    new_turn = InterviewTurn(session_id=session.id, question=next_question, answer_text=None)
    db.add(new_turn)
    db.flush()
    session.result_json = build_in_progress_result_json(session, config, next_question)
    session.result_json["passes_used"] = passes_used + 1
    session.result_json["asked_topics"] = (asked_topics + [extract_topic_hint(next_question)])[-20:]
    session.result_json["recent_avoid_questions"] = avoid_questions[-30:]
    session.result_json["attempts_by_turn"] = dict(session_json.get("attempts_by_turn", {}))
    session.result_json["attempt_logs"] = list(session_json.get("attempt_logs", []))
    session.result_json["question_rag"] = question_rag
    db.add(session)
    db.commit()

    return {
        "next_question": next_question,
        "pending_next_question": None,
        "feedback": f"Pass applied. Remaining passes: {MAX_PASSES_PER_SESSION - (passes_used + 1)}.",
        "score": 0,
        "done": False,
        "can_retry": False,
        "attempts_left": 0,
        "confidence_score": 40,
        "red_flags": ["question_skipped"],
        "score_explanation": "Skipped answers reduce evaluation quality and lower confidence in final scoring.",
        "passes_left": MAX_PASSES_PER_SESSION - (passes_used + 1),
        "question_index": len(answered_turns) + 1,
        "total_questions": max_turns,
        "sub_scores": {k: 0 for k in RUBRIC_KEYS},
        "scorecard": {},
        "tone_signals": {},
        "company_rubric": {},
        "strengths": [],
        "weaknesses": ["Current question was skipped."],
        "suggestions": ["Use remaining passes carefully and answer next question in detail."],
        "recommended_next_steps": [],
        "retrieval_evidence": [],
        "question_context": _session_question_context(session),
    }


def evaluate_reliability_for_session(
    db: Session,
    user: User,
    session_id: int,
    answer_text: str,
    runs: int = 3,
) -> dict[str, Any]:
    session = db.get(InterviewSession, session_id)
    if not session or session.user_id != user.id:
        raise ValueError("Session not found")
    turns = sorted(session.turns, key=lambda t: t.id)
    if not turns:
        raise ValueError("No turns found")
    question = turns[-1].question
    config = normalize_config((session.result_json or {}).get("config", {}))
    previous_turns = serialize_answered_turns([t for t in turns if t.answer_text is not None])
    safe_runs = max(2, min(6, int(runs)))
    evaluations = [
        evaluate_answer(
            session.profession,
            question,
            answer_text,
            config=config,
            previous_turns=previous_turns,
            attempt_index=1,
            use_rag=True,
        )
        for _ in range(safe_runs)
    ]
    scores = [int(item.get("score", 0)) for item in evaluations]
    reliability = score_reliability(scores)
    return {
        "question": question,
        "profession": session.profession,
        "runs": safe_runs,
        "reliability": reliability,
        "samples": [
            {
                "score": item.get("score"),
                "score_explanation": item.get("score_explanation"),
                "feedback": item.get("feedback"),
                "confidence_score": item.get("confidence_score"),
            }
            for item in evaluations
        ],
    }


def compare_rag_modes_for_session(
    db: Session,
    user: User,
    session_id: int,
    answer_text: str,
) -> dict[str, Any]:
    session = db.get(InterviewSession, session_id)
    if not session or session.user_id != user.id:
        raise ValueError("Session not found")
    turns = sorted(session.turns, key=lambda t: t.id)
    if not turns:
        raise ValueError("No turns found")
    question = turns[-1].question
    config = normalize_config((session.result_json or {}).get("config", {}))
    previous_turns = serialize_answered_turns([t for t in turns if t.answer_text is not None])
    with_rag = evaluate_answer(
        session.profession,
        question,
        answer_text,
        config=config,
        previous_turns=previous_turns,
        attempt_index=1,
        use_rag=True,
    )
    without_rag = evaluate_answer(
        session.profession,
        question,
        answer_text,
        config=config,
        previous_turns=previous_turns,
        attempt_index=1,
        use_rag=False,
    )
    score_delta = int(with_rag.get("score", 0)) - int(without_rag.get("score", 0))
    confidence_delta = int(with_rag.get("confidence_score", 0)) - int(without_rag.get("confidence_score", 0))
    preferred = "rag" if score_delta >= 0 else "no_rag"
    return {
        "question": question,
        "profession": session.profession,
        "preferred_mode": preferred,
        "score_delta": score_delta,
        "confidence_delta": confidence_delta,
        "with_rag": {
            "score": with_rag.get("score"),
            "confidence_score": with_rag.get("confidence_score"),
            "score_explanation": with_rag.get("score_explanation"),
            "feedback": with_rag.get("feedback"),
            "retrieval_evidence": with_rag.get("retrieval_evidence", []),
            "rag_summary": with_rag.get("rag_summary"),
            "retrieval_quality": with_rag.get("retrieval_quality"),
            "citations": with_rag.get("citations", []),
            "rag_evaluation": with_rag.get("rag_evaluation", {}),
        },
        "without_rag": {
            "score": without_rag.get("score"),
            "confidence_score": without_rag.get("confidence_score"),
            "score_explanation": without_rag.get("score_explanation"),
            "feedback": without_rag.get("feedback"),
            "retrieval_evidence": [],
        },
    }