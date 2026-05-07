from fastapi.testclient import TestClient
from uuid import uuid4

from app.main import app
import app.interview as interview_module


client = TestClient(app)


def _register_and_login(email: str) -> dict[str, str]:
    reg = client.post(
        "/auth/register",
        json={
            "name": "Test User",
            "email": email,
            "password": "secret123",
            "profession": "Frontend Developer",
        },
    )
    assert reg.status_code == 200
    data = reg.json()
    assert data["access_token"]
    assert data["csrf_token"]
    return {
        "Authorization": f"Bearer {data['access_token']}",
        "X-CSRF-Token": data["csrf_token"],
    }


def test_auth_start_and_submit_text_flow(monkeypatch):
    def fake_eval(*args, **kwargs):
        return {
            "score": 90,
            "sub_scores": {
                "communication": 80,
                "technical_depth": 75,
                "confidence": 76,
                "clarity": 79,
                "structure": 77,
                "problem_solving": 81,
            },
            "strengths": ["clear story"],
            "weaknesses": ["needs one metric"],
            "suggestions": ["add measurable impact"],
            "recommended_next_steps": ["practice 2 more answers"],
            "feedback": "Solid answer.",
            "next_question": "What trade-off did you make recently?",
            "done": False,
            "retrieval_evidence": [],
            "red_flags": [],
            "confidence_score": 72,
            "score_explanation": "Strong communication and structure.",
        }

    monkeypatch.setattr(interview_module, "evaluate_answer", fake_eval)

    headers = _register_and_login(f"flow_test_{uuid4().hex[:8]}@example.com")

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
    assert session_id > 0

    answer = client.post(
        "/interview/answer/text",
        json={"session_id": session_id, "answer_text": "I improved page speed and reduced load times by 25%."},
        headers=headers,
    )
    assert answer.status_code == 200
    payload = answer.json()
    assert payload["score"] == 90
    assert payload["score_explanation"]
    assert payload["next_question"] == "What trade-off did you make recently?"
