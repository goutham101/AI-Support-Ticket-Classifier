"""
analyze_errors.py

Finds the model's worst mistakes: cases where it was confidently wrong, not
just wrong. A model that's 34% sure and wrong is expected noise near a
4-way coin flip; a model that's 90% sure and wrong is confidently broken on
that input, and those are the errors worth reading.

Reuses train.py's exact model-selection logic (same data split, same CV,
same winner-picking rule) so this always analyzes whichever model train.py
actually chose, not whatever happened to win last time someone looked. Only
touches the dev set (via cross-validated predictions) -- the held-out test
set stays untouched, same discipline as train.py.

Run:
    python analyze_errors.py
"""

import sys
from pathlib import Path

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

N_WORST = 30


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    df = load_data(data_path)

    x = df["ticket_text"]
    y = df["category"]

    # Same split as train.py -- same random_state, same held-out set left alone.
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

    predicted_idx = y_proba.argmax(axis=1)
    predicted_label = [labels[i] for i in predicted_idx]
    confidence = y_proba.max(axis=1)

    errors = pd.DataFrame({
        "ticket_text": x_dev.values,
        "true_label": y_dev.values,
        "predicted_label": predicted_label,
        "confidence": confidence,
    })
    errors = errors[errors["true_label"] != errors["predicted_label"]]
    errors = errors.sort_values("confidence", ascending=False).head(N_WORST)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "worst_errors.csv"
    errors.to_csv(out_path, index=False)

    print(f"Model analyzed: {chosen_name}")
    print(f"{len(errors)} worst misclassifications (most confidently wrong first):\n")
    for _, row in errors.iterrows():
        print(f"[{row['confidence']:.3f}] true={row['true_label']!r} pred={row['predicted_label']!r}")
        print(f"    {row['ticket_text']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
