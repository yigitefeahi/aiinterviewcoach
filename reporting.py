from typing import Any

from .models import InterviewSession


def build_session_report(session: InterviewSession) -> dict[str, Any]:
    result_json = session.result_json or {}
    summary = result_json.get("final_summary") or {}
    turns = result_json.get("turns") or []
    turn_scores = [int(t.get("score", 0)) for t in turns if isinstance(t, dict) and t.get("score") is not None]
    first_half = turn_scores[: max(1, len(turn_scores) // 2)] if turn_scores else []
    second_half = turn_scores[max(1, len(turn_scores) // 2):] if turn_scores else []
    early_avg = int(sum(first_half) / len(first_half)) if first_half else None
    late_avg = int(sum(second_half) / len(second_half)) if second_half else None
    delta = (late_avg - early_avg) if (early_avg is not None and late_avg is not None) else None
    trend = "stable"
    if delta is not None:
        if delta >= 8:
            trend = "improving"
        elif delta <= -8:
            trend = "declining"

    return {
        "session_id": session.id,
        "profession": session.profession,
        "created_at": str(session.created_at),
        "overall_score": summary.get("score", result_json.get("average_score")),
        "rubric": summary.get("sub_scores", {}),
        "strengths": summary.get("strengths", []),
        "weaknesses": summary.get("weaknesses", []),
        "recommended_next_steps": summary.get("recommended_next_steps", []),
        "retrieval_evidence": summary.get("retrieval_evidence", []),
        "confidence_score": summary.get("confidence_score"),
        "red_flags": summary.get("red_flags", []),
        "turn_count": len(turns),
        "status": result_json.get("status"),
        "benchmark": {
            "early_average_score": early_avg,
            "late_average_score": late_avg,
            "score_delta": delta,
            "trend": trend,
            "retry_attempt_count": len(result_json.get("attempt_logs", []) or []),
        },
    }


def build_regression_snapshot(sessions: list[InterviewSession]) -> dict[str, Any]:
    completed_scores: list[int] = []
    confidence_scores: list[int] = []
    red_flag_counts: list[int] = []
    for s in sessions:
        result_json = s.result_json or {}
        if result_json.get("status") != "completed":
            continue
        summary = result_json.get("final_summary") or {}
        completed_scores.append(int(summary.get("score", result_json.get("average_score") or 0)))
        confidence_scores.append(int(summary.get("confidence_score", 0) or 0))
        red_flag_counts.append(len(summary.get("red_flags", []) or []))

    if not completed_scores:
        return {"status": "insufficient_data", "sample_size": 0}

    avg_score = int(sum(completed_scores) / len(completed_scores))
    avg_conf = int(sum(confidence_scores) / len(confidence_scores))
    avg_red_flags = round(sum(red_flag_counts) / len(red_flag_counts), 2)
    status = "pass" if (avg_score >= 55 and avg_conf >= 55 and avg_red_flags <= 3) else "fail"
    return {
        "status": status,
        "sample_size": len(completed_scores),
        "avg_score": avg_score,
        "avg_confidence": avg_conf,
        "avg_red_flags": avg_red_flags,
    }
