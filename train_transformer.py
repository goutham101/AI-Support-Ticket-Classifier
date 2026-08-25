"""
train_transformer.py

Phase 5: compares the deployed TF-IDF + Logistic Regression pipeline against
sentence embeddings (all-MiniLM-L6-v2) + a Logistic Regression head, using the
identical dev/held-out split and CV protocol as train.py so the comparison is
apples-to-apples, not two different experiments.

The point of this script isn't "which model wins" -- it's the tradeoff table
at the end: F1 vs. latency vs. model size. Confirmed with the project owner
before installing sentence-transformers (and its torch/transformers
dependencies), per the project's ground rules on heavy dependencies.

Run:
    python train_transformer.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from train import (
    DEFAULT_DATA_PATH,
    HOLDOUT_SIZE,
    MODEL_PATH,
    N_FOLDS,
    RANDOM_STATE,
    REPORTS_DIR,
    load_data,
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TRANSFORMER_HEAD_PATH = Path(__file__).parent / "transformer_head.joblib"
LATENCY_SAMPLES = 100


def measure_latency(predict_one, texts) -> float:
    """Time single-item predictions one at a time (not batched -- that's what
    a live /classify request actually looks like) and return the median, in ms."""
    latencies = []
    for text in texts:
        t0 = time.perf_counter()
        predict_one(text)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.median(latencies))


def main():
    df = load_data(DEFAULT_DATA_PATH)
    x = df["ticket_text"].tolist()
    y = df["category"]

    # Same split as train.py: same random_state, same held-out set, touched once.
    x_dev, x_holdout, y_dev, y_holdout = train_test_split(
        x, y, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Encoding {len(x_dev)} dev tickets...")
    t0 = time.time()
    x_dev_emb = embedder.encode(x_dev, show_progress_bar=True, batch_size=64)
    print(f"Encoded dev set in {time.time() - t0:.1f}s")

    print(f"Encoding {len(x_holdout)} held-out tickets...")
    x_holdout_emb = embedder.encode(x_holdout, show_progress_bar=True, batch_size=64)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    head = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(head, x_dev_emb, y_dev, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"\nEMBEDDING + LOGISTIC_REGRESSION macro-F1 (5-fold CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    head.fit(x_dev_emb, y_dev)
    y_pred_holdout = head.predict(x_holdout_emb)
    holdout_f1 = f1_score(y_holdout, y_pred_holdout, average="macro")
    print(f"Held-out macro-F1 ({len(x_holdout)} samples): {holdout_f1:.4f}")

    joblib.dump(head, TRANSFORMER_HEAD_PATH)
    print(f"Saved transformer head to {TRANSFORMER_HEAD_PATH}")

    # Latency: same 100 held-out tickets for both approaches, one prediction at a time.
    latency_texts = x_holdout[:LATENCY_SAMPLES]

    def transformer_predict_one(text):
        emb = embedder.encode([text])
        return head.predict(emb)

    transformer_latency_ms = measure_latency(transformer_predict_one, latency_texts)
    print(f"\nTransformer median latency ({LATENCY_SAMPLES} single predictions): {transformer_latency_ms:.2f} ms/ticket")

    tfidf_pipeline = joblib.load(MODEL_PATH)

    def tfidf_predict_one(text):
        return tfidf_pipeline.predict([text])

    tfidf_latency_ms = measure_latency(tfidf_predict_one, latency_texts)
    print(f"TF-IDF + LogReg median latency ({LATENCY_SAMPLES} single predictions): {tfidf_latency_ms:.2f} ms/ticket")

    tfidf_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    transformer_head_size_mb = TRANSFORMER_HEAD_PATH.stat().st_size / (1024 * 1024)

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--sentence-transformers--{EMBEDDING_MODEL_NAME}"
    embedding_model_size_mb = None
    if hf_cache.exists():
        # HF's cache stores real files under blobs/ and symlinks them into
        # snapshots/ -- rglob resolves both, so dedupe by resolved path or
        # every file gets counted twice.
        unique_files = {f.resolve() for f in hf_cache.rglob("*") if f.is_file()}
        embedding_model_size_mb = sum(f.stat().st_size for f in unique_files) / (1024 * 1024)

    print("\nMODEL SIZE")
    print(f"TF-IDF + LogReg (best_ticket_classifier.joblib): {tfidf_size_mb:.2f} MB")
    print(f"Transformer head (transformer_head.joblib): {transformer_head_size_mb:.2f} MB")
    if embedding_model_size_mb is not None:
        print(f"all-MiniLM-L6-v2 weights (HF cache): {embedding_model_size_mb:.1f} MB")
        print(f"Transformer approach total: {transformer_head_size_mb + embedding_model_size_mb:.1f} MB")

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "transformer_comparison.txt", "w") as f:
        f.write(f"embedding_cv_f1_mean={cv_scores.mean():.4f}\n")
        f.write(f"embedding_cv_f1_std={cv_scores.std():.4f}\n")
        f.write(f"embedding_holdout_f1={holdout_f1:.4f}\n")
        f.write(f"transformer_latency_ms={transformer_latency_ms:.2f}\n")
        f.write(f"tfidf_latency_ms={tfidf_latency_ms:.2f}\n")
        f.write(f"tfidf_size_mb={tfidf_size_mb:.2f}\n")
        f.write(f"transformer_head_size_mb={transformer_head_size_mb:.2f}\n")
        if embedding_model_size_mb is not None:
            f.write(f"embedding_model_size_mb={embedding_model_size_mb:.1f}\n")
    print(f"\nSaved raw numbers to {REPORTS_DIR / 'transformer_comparison.txt'}")


if __name__ == "__main__":
    main()
