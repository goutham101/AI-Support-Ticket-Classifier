"""
main.py

FastAPI wrapper around the ticket classifier.

Run locally:
    uvicorn main:app --reload
Then open:
    http://127.0.0.1:8000/static/index.html   (the demo UI)
    http://127.0.0.1:8000/docs                (auto-generated API docs)
"""

from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

MODEL_PATH = Path(__file__).parent / "best_ticket_classifier.joblib"

# Picked with select_threshold.py on 2026-08-04: the smallest threshold where
# precision on the *accepted* subset (predictions confident enough to keep)
# exceeds 95%, measured on cross-validated dev-set predictions (never the
# held-out test set). At 0.61: precision 0.9505, coverage 0.7344 -- about a
# quarter of tickets fall below this and get routed to a human instead of a
# guess. See reports/threshold_sweep.csv for the full sweep.
CONFIDENCE_THRESHOLD = 0.61
TOP_K = 3


def is_low_confidence(confidence: float) -> bool:
    return confidence < CONFIDENCE_THRESHOLD

app = FastAPI(
    title="Support Ticket Classifier API",
    description="Classifies banking support tickets into 77 fine-grained categories.",
    version="1.0.0",
)

# Serve the tiny frontend at /static/index.html
# NOTE: this must point at a dedicated static/ folder, never the project
# root -- mounting the root would expose train.py, the dataset, the model
# file, and everything else in this directory to anyone who guesses a path.
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Load the trained model once at startup, not per-request
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None  # API still boots; /classify will explain what to do


class TicketRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > 5000:
            raise ValueError("text must be under 5000 characters")
        return v.strip()


class TopKPrediction(BaseModel):
    category: str
    probability: float


class TicketResponse(BaseModel):
    category: str
    confidence: float
    needs_review: bool
    top_k: List[TopKPrediction]


from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/classify", response_model=TicketResponse)
def classify(request: TicketRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` first to generate best_ticket_classifier.joblib.",
        )

    try:
        # Not every sklearn estimator supports predict_proba (e.g. some SVMs),
        # but every model this project trains does -- if that ever changes,
        # fail loudly instead of silently reporting fake confidence.
        proba = model.predict_proba([request.text])[0]
        ranked = sorted(zip(model.classes_, proba), key=lambda pair: pair[1], reverse=True)
    except Exception as exc:
        # Never leak internals to the client; log server-side in a real deployment.
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    top_label, top_confidence = ranked[0]
    needs_review = is_low_confidence(top_confidence)
    top_k = [
        TopKPrediction(category=label, probability=round(float(prob), 4))
        for label, prob in ranked[:TOP_K]
    ]

    return TicketResponse(
        category="needs_review" if needs_review else top_label,
        confidence=round(float(top_confidence), 4),
        needs_review=needs_review,
        top_k=top_k,
    )
