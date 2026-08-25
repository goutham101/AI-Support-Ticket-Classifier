"""
select_threshold.py

Picks the confidence threshold used by /classify to decide when to return
"needs_review" instead of a guess. Picked empirically, not hard-coded: sweep
threshold values and find the smallest one where precision on the *accepted*
subset (predictions the model was confident enough to make) exceeds 95%.

Uses cross-validated predictions on the dev set only -- same discipline as
everywhere else in this project, the held-out test set is never used to pick
a threshold. Tuning a threshold on the same data you report your final score
on would make that score dishonest.

Run:
    python select_threshold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from train import (
    DEFAULT_DATA_PATH,
    HOLDOUT_SIZE,
    N_FOLDS,
    RANDOM_STATE,
    REPORTS_DIR,
    build_models,
    load_data,
    run_cross_validation,
    select_model,
)

PRECISION_TARGET = 0.95
THRESHOLDS = np.arange(0.0, 1.0, 0.01)


def main():
    df = load_data(DEFAULT_DATA_PATH)
    x = df["ticket_text"]
    y = df["category"]

    x_dev, _x_holdout, y_dev, _y_holdout = train_test_split(
        x, y, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = build_models()
    results = run_cross_validation(models, x_dev, y_dev, cv)
    chosen_name = select_model(models, results)
    chosen_pipeline = models[chosen_name]

    labels = sorted(y_dev.unique())
    y_proba = cross_val_predict(
        chosen_pipeline, x_dev, y_dev, cv=cv, method="predict_proba", n_jobs=-1
    )
    predicted_label = np.array([labels[i] for i in y_proba.argmax(axis=1)])
    confidence = y_proba.max(axis=1)
    correct = predicted_label == y_dev.values

    rows = []
    chosen_threshold = None
    for t in THRESHOLDS:
        accepted = confidence >= t
        coverage = accepted.mean()
        precision = correct[accepted].mean() if accepted.any() else float("nan")
        rows.append({"threshold": round(float(t), 2), "coverage": coverage, "precision": precision})
        if chosen_threshold is None and accepted.any() and precision > PRECISION_TARGET:
            chosen_threshold = round(float(t), 2)

    sweep = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    sweep.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)

    print(f"Model: {chosen_name}")
    print(f"Target: precision > {PRECISION_TARGET:.0%} on the accepted subset\n")
    print(sweep[sweep["threshold"] % 0.1 < 1e-9].to_string(index=False))

    if chosen_threshold is None:
        print(f"\nNo threshold in the sweep reached {PRECISION_TARGET:.0%} precision.")
    else:
        row = sweep[sweep["threshold"] == chosen_threshold].iloc[0]
        print(f"\nCHOSEN THRESHOLD: {chosen_threshold}")
        print(f"Coverage at this threshold: {row['coverage']:.4f} ({row['coverage']*len(x_dev):.0f} of {len(x_dev)} dev tickets accepted)")
        print(f"Precision at this threshold: {row['precision']:.4f}")
        print(f"\nSaved full sweep to {REPORTS_DIR / 'threshold_sweep.csv'}")


if __name__ == "__main__":
    main()
