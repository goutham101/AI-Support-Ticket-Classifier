# Support Ticket Classifier API

Classifies unstructured support ticket text into categories (billing, technical_issue,
account_access, feature_request) using a TF-IDF + Naive Bayes / Logistic Regression
pipeline, served via FastAPI with a minimal web UI.

**Live demo:** https://ticket-classifier-gbus.onrender.com/static/index.html


**Screenshot:** 
<img width="1482" height="809" alt="image" src="https://github.com/user-attachments/assets/c246bdad-938c-4652-aa65-543aad04868c" />



## Why this exists

Manually triaging support tickets doesn't scale. This project trains a text classification
model on labeled ticket data, automatically selects the better-performing algorithm
(Naive Bayes vs. Logistic Regression), and exposes it as a REST API with a web UI so a
non-technical user can paste a ticket and get an instant category prediction.

## Architecture

```
Browser (static/index.html)
        |
        v
POST /classify  --->  FastAPI (main.py)  --->  best_ticket_classifier.joblib
                                                  (trained by train.py on support_tickets.csv)
```

`static/` is the only directory the API serves over HTTP — it contains just
`index.html`. `main.py` mounts `static/` specifically rather than the project root,
so the training script, dataset, and model file are never web-accessible.

- **train.py** — loads `support_tickets.csv`, trains both Naive Bayes and Logistic
  Regression pipelines, evaluates both, and saves whichever scores higher accuracy
  (selection is by accuracy alone — an earlier version of this file broke ties by
  average model confidence, which quietly rewarded an overconfident model over a
  more accurate one; that logic was removed)
- **main.py** — FastAPI app; loads the model once at startup, exposes `/classify` and `/health`
- **static/index.html** — single-page UI: text box, button, displays category + confidence
- **predict.py** — original CLI entry point (kept for local one-off testing)
- **test_main.py** — pytest suite covering valid input, empty input, missing fields, oversized input

## Running locally

```bash
pip install -r requirements.txt
python train.py                # trains both models, saves the better one
uvicorn main:app --reload      # starts the API on http://127.0.0.1:8000
```

Then open `http://127.0.0.1:8000/static/index.html` in a browser, or hit the API directly:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My app keeps crashing when I open settings"}'
```

Response:

```json
{"category": "technical_issue", "confidence": 0.4051}
```

## Running tests

```bash
pytest
```

## Model performance (current dataset: 244 tickets)

```
NAIVE_BAYES           accuracy: 0.885  (macro F1: 0.89)
LOGISTIC_REGRESSION   accuracy: 0.869  (macro F1: 0.87)
Best model selected: naive_bayes
```

Per-category breakdown (Naive Bayes, the selected model):

| Category        | Precision | Recall | F1   |
|------------------|-----------|--------|------|
| billing          | 0.94      | 1.00   | 0.97 |
| feature_request  | 1.00      | 0.87   | 0.93 |
| account_access   | 0.81      | 0.87   | 0.84 |
| technical_issue  | 0.81      | 0.81   | 0.81 |

Verified against ticket phrasing written specifically to differ from both the training
data and the held-out test set (not just a reworded training example) — 5/5 correctly
classified, with confidence scores of 0.34-0.57 (baseline random guessing across 4
categories is 0.25, so this is meaningfully above chance).

**Known limitations:**
- `account_access` and `technical_issue` are the weakest categories (F1 0.81-0.84),
  likely due to overlapping vocabulary (e.g. "can't access" / "won't load" language
  appears in both).
- The dataset is currently slightly imbalanced: `technical_issue` has 64 examples vs.
  60 for the other three categories. Not large enough to bias results meaningfully at
  this scale, but worth re-balancing if the dataset grows further.
- The dataset is synthetic/AI-assisted rather than sourced from real user tickets — see
  `build_dataset.py`. Before calling this production-quality: (1) review every row for
  realism, (2) grow past current counts if possible, (3) consider adding a fifth
  "unclear" category for genuinely ambiguous cases.

## Deployment

Deployed on [Render / Railway / Hugging Face Spaces — pick one] using the included
`requirements.txt`. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.

## Possible extensions

- Expand the dataset (see limitation above) and retrain
- Batch classification endpoint (`POST /classify/batch`)
- Log low-confidence predictions for human review (confidence < 0.6)
- Auth + rate limiting if exposed publicly long-term
