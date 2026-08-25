"""
test_main.py

Run:
    pytest

These are intentionally simple — the point isn't test sophistication,
it's proving to a reader (and yourself) that the API handles bad input
without crashing.
"""

from fastapi.testclient import TestClient
from main import CONFIDENCE_THRESHOLD, app, is_low_confidence

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


def test_classify_low_confidence_needs_review():
    # Deliberately vague -- no banking vocabulary for any of the 77 classes
    # to grab onto, so confidence should land below CONFIDENCE_THRESHOLD.
    response = client.post("/classify", json={"text": "hello I have a question about something"})
    assert response.status_code == 200
    body = response.json()
    assert body["needs_review"] is True
    assert body["category"] == "needs_review"
    assert body["confidence"] < CONFIDENCE_THRESHOLD


def test_classify_top_k_sorted_and_bounded():
    response = client.post("/classify", json={"text": "I was charged twice this month"})
    assert response.status_code == 200
    top_k = response.json()["top_k"]

    assert len(top_k) == 3
    probabilities = [item["probability"] for item in top_k]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1.0 + 1e-9
    assert all(0.0 <= p <= 1.0 for p in probabilities)


def test_confidence_threshold_boundary():
    assert is_low_confidence(CONFIDENCE_THRESHOLD - 0.0001) is True
    assert is_low_confidence(CONFIDENCE_THRESHOLD) is False
    assert is_low_confidence(CONFIDENCE_THRESHOLD + 0.0001) is False
