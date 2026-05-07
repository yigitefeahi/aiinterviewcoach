"""Golden-file style checks: mocked LLM scores must land in documented ranges."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app
import app.interview as interview_module


client = TestClient(app)


def _headers(reg_json: dict) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {reg_json['access_token']}",
        "X-CSRF-Token": reg_json["csrf_token"],
    }


def test_submit_answer_score_within_expected_range(monkeypatch):
    """When evaluate_answer returns 72, response score matches (pipeline regression)."""

    def fake_eval(*args, **kwargs):
        return {
            "score": 72,
            "sub_scores": {
                "communication": 70,
                "technical_depth": 71,
                "confidence": 69,
                "clarity": 72,
                "structure": 70,
                "problem_solving": 73,
            },
            "strengths": ["ok"],
            "weaknesses": ["more metrics"],
            "suggestions": ["add KPI"],
            "recommended_next_steps": ["practice"],
            "feedback": "Solid.",
            "next_question": "Next?",
            "done": False,
            "retrieval_evidence": [],
            "red_flags": [],
            "confidence_score": 70,
            "score_explanation": "ok",
        }

    monkeypatch.setattr(interview_module, "evaluate_answer", fake_eval)

    reg = client.post(
        "/auth/register",
        json={
            "name": "Golden",
            "email": f"gold_{uuid4().hex[:8]}@example.com",
            "password": "secret123",
            "profession": "Frontend Developer",
        },
    )
    assert reg.status_code == 200
    h = _headers(reg.json())

    start = client.post(
        "/interview/start",
        json={
            "profession": "Frontend Developer",
            "difficulty": "Junior",
            "mode": "text",
            "interview_length": "5 Questions",
            "focus_area": "Mixed",
        },
        headers=h,
    )
    assert start.status_code == 200
    sid = start.json()["session_id"]

    ans = client.post(
        "/interview/answer/text",
        json={"session_id": sid, "answer_text": "x" * 30},
        headers=h,
    )
    assert ans.status_code == 200
    assert ans.json()["score"] == 72
