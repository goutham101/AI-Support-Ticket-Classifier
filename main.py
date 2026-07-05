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

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

MODEL_PATH = Path(__file__).parent / "best_ticket_classifier.joblib"

app = FastAPI(
    title="Support Ticket Classifier API",
    description="Classifies support ticket text into billing, technical, or account_access.",
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


class TicketResponse(BaseModel):
    category: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Support Ticket Classifier API. See /docs for usage."}


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
        prediction = model.predict([request.text])[0]
        # Not every sklearn estimator supports predict_proba (e.g. some SVMs).
        # Guard so the endpoint never crashes on that.
        if hasattr(model, "predict_proba"):
            confidence = float(max(model.predict_proba([request.text])[0]))
        else:
            confidence = 1.0
    except Exception as exc:
        # Never leak internals to the client; log server-side in a real deployment.
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return TicketResponse(category=prediction, confidence=round(confidence, 4))
