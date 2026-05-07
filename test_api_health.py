from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_health_deps_shape():
    r = client.get("/health/deps")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["jwt_backend"] == "PyJWT"
    assert "chroma_dir_writable" in data
    assert "database_scheme" in data
    assert isinstance(data.get("hints"), list)
