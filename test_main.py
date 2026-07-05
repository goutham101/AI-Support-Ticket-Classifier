"""
test_main.py

Run:
    pytest

These are intentionally simple — the point isn't test sophistication,
it's proving to a reader (and yourself) that the API handles bad input
without crashing.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classify_valid_ticket():
    response = client.post("/classify", json={"text": "I was charged twice this month"})
    assert response.status_code == 200
    body = response.json()
    assert "category" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_classify_empty_text_rejected():
    response = client.post("/classify", json={"text": "   "})
    assert response.status_code == 422  # pydantic validation error


def test_classify_missing_field_rejected():
    response = client.post("/classify", json={})
    assert response.status_code == 422


def test_classify_oversized_text_rejected():
    response = client.post("/classify", json={"text": "a" * 6000})
    assert response.status_code == 422
