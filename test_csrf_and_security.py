"""CSRF + origin checks for authenticated mutating requests."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_post_without_csrf_returns_403_when_authenticated():
    reg = client.post(
        "/auth/register",
        json={
            "name": "CSRF Test",
            "email": f"csrf_{uuid4().hex[:8]}@example.com",
            "password": "secret123",
            "profession": "Frontend Developer",
        },
    )
    assert reg.status_code == 200
    data = reg.json()
    headers = {"Authorization": f"Bearer {data['access_token']}"}
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
    assert start.status_code == 403


def test_post_with_csrf_succeeds():
    reg = client.post(
        "/auth/register",
        json={
            "name": "CSRF OK",
            "email": f"csrfok_{uuid4().hex[:8]}@example.com",
            "password": "secret123",
            "profession": "Frontend Developer",
        },
    )
    assert reg.status_code == 200
    data = reg.json()
    headers = {
        "Authorization": f"Bearer {data['access_token']}",
        "X-CSRF-Token": data["csrf_token"],
    }
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
