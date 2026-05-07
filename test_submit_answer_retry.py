"""Regression: retry path must not reference undefined locals (e.g. question_index / max_turns)."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
import app.interview as interview_module


client = TestClient(app)


def _register_and_login(email: str) -> dict[str, str]:
    reg = client.post(
        "/auth/register",
        json={
            "name": "Retry Test",
            "email": email,
            "password": "secret123",
            "profession": "Frontend Developer",
        },
    )
    assert reg.status_code == 200
    data = reg.json()
    return {
        "Authorization": f"Bearer {data['access_token']}",
        "X-CSRF-Token": data["csrf_token"],
    }


def _base_eval(score: int, done: bool = False):
    return {
        "score": score,
        "sub_scores": {
            "communication": 60,
            "technical_depth": 55,
            "confidence": 58,
            "clarity": 57,
            "structure": 56,
            "problem_solving": 59,
        },
        "strengths": ["some clarity"],
        "weaknesses": ["needs metrics"],
        "suggestions": ["add numbers"],
        "recommended_next_steps": ["practice"],
        "feedback": "Could be stronger.",
        "next_question": "Tell me about a conflict you resolved.",
        "done": done,
        "retrieval_evidence": [],
        "red_flags": [],
        "confidence_score": 50,
        "score_explanation": "Below threshold for retry.",
    }


def test_submit_answer_retry_branch_no_crash(monkeypatch):
    """Low score + attempts left → can_retry; response must include question_index / total_questions."""

    monkeypatch.setattr(interview_module, "evaluate_answer", lambda *a, **k: _base_eval(70))

    headers = _register_and_login(f"retry_{uuid4().hex[:10]}@example.com")

    start = client.post(
        "/interview/start",
        json={
            "profession": "Frontend Developer",
            "difficulty": "Junior",
            "mode": "text",
            "interview_length": "5 Questions",
            "focus_area": "Mixed",
        },
        headers=headers,
    )
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    answer = client.post(
        "/interview/answer/text",
        json={
            "session_id": session_id,
            "answer_text": "x" * 25,
        },
        headers=headers,
    )
    assert answer.status_code == 200
    payload = answer.json()
    assert payload["can_retry"] is True
    assert payload["attempts_left"] >= 0
    assert payload["question_index"] == 1
    assert payload["total_questions"] == 5
    assert "feedback" in payload
