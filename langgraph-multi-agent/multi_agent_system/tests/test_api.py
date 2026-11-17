from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_blank_question_is_rejected():
    response = client.post("/api/v1/ask", json={"question": "   "})
    assert response.status_code == 422


def test_missing_question_is_rejected():
    response = client.post("/api/v1/ask", json={})
    assert response.status_code == 422
