import pytest


@pytest.fixture(autouse=True)
def disable_cv_embedding_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid OpenAI embedding calls during unit/integration tests."""
    from app.config import settings

    monkeypatch.setattr(settings, "cv_use_embedding", False)
    monkeypatch.setattr(settings, "cv_llm_screening", False)


def auth_headers_from_register(reg_json: dict) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {reg_json['access_token']}",
        "X-CSRF-Token": reg_json["csrf_token"],
    }
