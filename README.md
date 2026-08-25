# Support Ticket Classifier API

Classifies banking support tickets into 77 fine-grained categories using TF-IDF +
Logistic Regression, trained on the real [Banking77](https://arxiv.org/abs/2003.04807)
dataset and served via FastAPI with a minimal web UI.

**Live demo:** https://ticket-classifier-gbus.onrender.com/static/index.html


**Screenshot:** 
<img width="1482" height="809" alt="image" src="https://github.com/user-attachments/assets/c246bdad-938c-4652-aa65-543aad04868c" />



## Results

Real output from `python train.py`:

| Model | CV macro-F1 (5-fold, mean ± std) | Held-out macro-F1 |
|---|---|---|
| Baseline (most-frequent class) | 0.0004 ± 0.0000 | — |
| Naive Bayes (tuned) | 0.8129 ± 0.0134 | — |
| **Logistic Regression (tuned) — deployed** | **0.8423 ± 0.0079** | **0.8534** |

```
MODEL SELECTION
Mean macro-F1 difference: 0.0294
Threshold (larger of the two CV standard deviations): 0.0134
logistic_regression beats the other model by more than one standard deviation -> selecting logistic_regression.

FINAL HELD-OUT RESULT (touched once)
Model: logistic_regression
Held-out macro-F1 (1963 samples): 0.8534
```

**How the deployed model was selected:** stratified 5-fold CV on 85% of the data (the
other 15% is held out and never touched until the very end). Naive Bayes and Logistic
Regression are compared on CV mean macro-F1, and a winner is only declared if the gap
exceeds one standard deviation — otherwise the difference is noise, and the script
defaults to the simpler model instead of pretending it found a winner.

Hyperparameters came from a `GridSearchCV` sweep in `tune.py` (searched on the dev split
only, never the held-out set), not guesswork:

| Model | Searched grid | Best params | CV macro-F1 |
|---|---|---|---|
| Naive Bayes | `ngram_range` ∈ {(1,1),(1,2)} × `min_df` ∈ {1,2,3} × `alpha` ∈ {0.1,0.5,1.0} | `alpha=0.1, min_df=1, ngram_range=(1,2)` | 0.8129 ± 0.0134 |
| Logistic Regression | `ngram_range` ∈ {(1,1),(1,2)} × `min_df` ∈ {1,2,3} × `C` ∈ {0.1,1,10} | `C=10, min_df=1, ngram_range=(1,1)` | 0.8423 ± 0.0079 |

Before tuning, logistic regression (the deployed model) scored 0.8379 ± 0.0116 CV /
0.8445 held-out; after, 0.8423 ± 0.0079 / 0.8534 — a modest gain, but the shrunk std
means it's also more consistent across folds. Its best config turned out to be unigrams
only; bigrams were a guess in the original hand-picked version, and CV says they don't
help once `min_df` and `C` are tuned properly.

A confusion matrix is at `reports/confusion_matrix.png`, and the 10 most-confused class
pairs at `reports/confused_pairs.csv`.

## Where the model fails

Run `python analyze_errors.py` for the 30 most *confidently wrong* predictions (high
confidence, wrong label) — saved to `reports/worst_errors.csv`.

Most-confused class pairs, from 5-fold CV predictions on the dev set:

| True label | Predicted label | Count | Why they likely confuse |
|---|---|---|---|
| `why_verify_identity` | `verify_my_identity` | 28 | Same vocabulary ("verify", "identity"); the real difference is *why* (a reason/complaint) vs. *how* (a task), which bag-of-words TF-IDF can't tell apart. |
| `verify_my_identity` | `why_verify_identity` | 20 | Same pair, other direction — confirms it's a genuine two-way overlap, not a one-off. |
| `top_up_by_bank_transfer_charge` | `transfer_fee_charged` | 18 | Both are about a fee on moving money; the two phrasings share almost all their n-grams once bigrams are dropped (the tuned model uses unigrams only). |
| `order_physical_card` | `getting_spare_card` | 15 | Both are about acquiring a physical card. Whether it's a first card or a replacement is often implicit, not stated. |
| `getting_spare_card` | `order_physical_card` | 14 | Same pair, other direction — a genuine two-way overlap, not a one-off. |

One concrete example from `reports/worst_errors.csv`:

```
Where are the cards transported to?
true=order_physical_card  pred=getting_spare_card  confidence=0.936
```

93.6% confident and wrong. "Transported to" reads like a delivery-logistics question,
which pulls it toward `getting_spare_card` (also about card delivery), even though the
ticket is really about getting a first card, not a spare.

**Known limitations:**
- The five pairs above account for a large share of total errors — confusion is
  concentrated in a handful of genuinely similar intents, not spread evenly across all
  77 classes.
- Several "worst errors" look like defensible relabels, not model failures — e.g. "How
  long will it take for me to get my card?" is labeled `card_arrival` but predicted
  `card_delivery_estimate`, and both readings are reasonable. Some ceiling on accuracy
  here comes from label ambiguity in the dataset itself, not just model weakness.
- The dataset is single-domain (UK/EU banking) and English-only, so none of these
  numbers say anything about how this would generalize to a different support domain.

## TF-IDF vs. sentence embeddings

Everything above is a bag-of-words model — it has no idea that "card hasn't shown up"
and "still waiting for my card" mean the same thing unless the words literally overlap.
`train_transformer.py` embeds each ticket with
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (87MB)
and trains a Logistic Regression head on the embeddings, using the same dev/held-out
split and CV protocol as the TF-IDF model so the comparison is fair.

Real output, latency measured over 100 single-ticket predictions (not batched — that's
what an actual `/classify` request looks like), median reported:

| Approach | Macro-F1 (CV) | Held-out F1 | Latency (ms/ticket) | Model size |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8423 ± 0.0079 | 0.8534 | 0.17 | 1.3 MB |
| all-MiniLM-L6-v2 + Logistic Regression | 0.9069 ± 0.0054 | 0.9100 | 5.74 | 87.5 MB |

The embedding approach wins on accuracy by a real margin (+0.057 held-out F1) — it
understands paraphrasing, which is exactly the `why_verify_identity` /
`verify_my_identity`-style confusion above that TF-IDF struggles with. But it's ~33x
slower per prediction and ~67x larger on disk. **TF-IDF stays the deployed model in
`main.py`**: it's simpler to ship, has no PyTorch dependency in production, and 0.85 F1
is already well above the 0.0004 baseline. Worth revisiting if this ever needs to run
somewhere accuracy matters more than deploy simplicity or 5ms/ticket is actually a
problem.

## Dataset

[Banking77](https://arxiv.org/abs/2003.04807) (PolyAI), CC BY 4.0 license. 13,083 real,
anonymized customer service queries about banking, labeled with 77 fine-grained intents
(`card_arrival`, `wrong_exchange_rate_for_cash_withdrawal`, etc). Class sizes range from
75 to 227 examples; 0% exact-duplicate ticket text. `load_data.py` fetches the two CSVs
PolyAI publishes (train + test splits) directly from their GitHub repo and combines them
— not via `datasets.load_dataset()`, because Hugging Face deprecated script-based
dataset loading and `PolyAI/banking77` still uses one; trying to load it with a current
`datasets` install throws `RuntimeError: Dataset scripts are no longer supported`.
Fetching the two CSVs the script itself points to sidesteps that, and avoids installing
`datasets` + `pyarrow` (~57MB) for two CSV files. Preprocessing is minimal: drop empty
rows, strip whitespace, and split off 15% as a held-out test set before any tuning or
cross-validation touches the rest.

**This project's original dataset was synthetic** — 244 rows I generated myself from
template pools across 4 keyword-separable categories (billing, technical_issue,
account_access, feature_request). That was close to a keyword-lookup task wearing an ML
costume: good enough to prove the API works, not good enough to say anything about
handling real, messy language. It's moved to `legacy/` (dataset + generator script) so
the original results stay reproducible, but nothing else in this repo uses it anymore.

## Setup / run instructions

```bash
pip install -r requirements.txt
python load_data.py            # downloads Banking77, caches to data/ (first time only)
python train.py                # trains both models, saves the better one
uvicorn main:app --reload      # starts the API on http://127.0.0.1:8000
```

Then open `http://127.0.0.1:8000/static/index.html` in a browser, or hit the API
directly:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "My card still has not arrived, it has been two weeks"}'
```

```json
{
  "category": "card_arrival",
  "confidence": 0.9437,
  "needs_review": false,
  "top_k": [
    {"category": "card_arrival", "probability": 0.9437},
    {"category": "reverted_card_payment?", "probability": 0.0205},
    {"category": "card_delivery_estimate", "probability": 0.0043}
  ]
}
```

A vague ticket the model can't confidently place gets flagged instead of guessed:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "hello I have a question about something"}'
```

```json
{
  "category": "needs_review",
  "confidence": 0.1465,
  "needs_review": true,
  "top_k": [
    {"category": "top_up_reverted", "probability": 0.1465},
    {"category": "top_up_failed", "probability": 0.0549},
    {"category": "top_up_limits", "probability": 0.0467}
  ]
}
```

**The 0.61 confidence threshold was picked on validation data, not test data.**
`select_threshold.py` sweeps threshold values against cross-validated dev-set
predictions and picks the smallest one where precision on the *accepted* subset exceeds
95% (at 0.61: 95.05% precision, 73.4% coverage). The held-out test set used for the
final score above is never touched when picking this threshold — tuning it on test data
would make both numbers dishonest. Full sweep in `reports/threshold_sweep.csv`.

Tests:

```bash
pytest                    # everything, 9 tests, ~1.4s
pytest -m "not slow"      # fast suite only (skips the CV regression test), ~1.2s
```

Deployment: deployed on Render (see live demo link at the top), start command
`uvicorn main:app --host 0.0.0.0 --port $PORT`. **Since `best_ticket_classifier.joblib`
is no longer committed (see Architecture below), Render's build command needs to
actually produce it** — e.g. `pip install -r requirements.txt && python load_data.py &&
python train.py` — or `/classify` will 503 with "Model not loaded" on a fresh deploy.
`requirements.txt` includes `sentence-transformers` (and `torch`) for
`train_transformer.py`'s comparison above, but `main.py` never imports it — if
disk/memory is tight on a free tier, those three packages can be dropped from the
deploy environment; nothing at request time needs them.

## Architecture

```
Browser (static/index.html)
        |
        v
POST /classify  --->  FastAPI (main.py)  --->  best_ticket_classifier.joblib
                                                  (trained by train.py on data/banking77.csv,
                                                   rebuilt by .github/workflows/train.yml on push)
```

`static/` is the only directory the API serves over HTTP — it contains just
`index.html`. `main.py` mounts `static/` specifically rather than the project root, so
the training script, dataset, and model file are never web-accessible.

- **load_data.py** — downloads Banking77, caches it to `data/banking77.csv` (gitignored)
- **train.py** — stratified 5-fold CV, baseline, model selection, held-out evaluation
  (see Results)
- **tune.py** — one-time `GridSearchCV` run; winning hyperparameters are hard-coded into
  `train.py`'s `build_models()`
- **analyze_errors.py** — dumps the 30 most confidently-wrong predictions (see Where the
  model fails)
- **train_transformer.py** — TF-IDF vs. sentence-embedding comparison (see above)
- **select_threshold.py** — picks the `/classify` confidence threshold empirically
- **main.py** — FastAPI app; loads the model once at startup, exposes `/classify` and
  `/health`; low-confidence predictions are flagged `needs_review` instead of guessed
- **static/index.html** — single-page UI: text box, button, shows the prediction or a
  "needs review" state with the top-3 candidates
- **predict.py** — thin CLI that POSTs to a running API (`uvicorn main:app` must already
  be up); doesn't load the model itself, so `main.py` is the only place prediction logic
  lives
- **test_main.py** — API-contract tests: valid/empty/missing/oversized input,
  low-confidence -> `needs_review`, `top_k` shape, threshold boundary
- **test_model.py** — CV regression test on `sample_tickets.csv`, marked
  `@pytest.mark.slow`
- **legacy/** — the original synthetic 244-row dataset and its generator script (see
  Dataset)
- **sample_tickets.csv** — 40 real Banking77 tickets (8 classes, 5 examples each), so
  the regression test can run stratified CV without downloading the full dataset
- **.github/workflows/train.yml** — retrains on every push to `main`, uploads
  `best_ticket_classifier.joblib` as a build artifact (the file itself isn't committed —
  a binary that changes every training run doesn't belong in git history)

## What I'd do next

- Batch classification endpoint (`POST /classify/batch`)
- Actually route `needs_review` tickets somewhere — right now the API flags them, but
  nothing downstream picks them up
- Auth + rate limiting if this ever gets exposed publicly long-term
- Revisit the TF-IDF/transformer tradeoff if this needs to run somewhere accuracy
  matters more than deploy simplicity
