"""
tune.py

Grid search over the TF-IDF pipeline hyperparameters. Runs once to find good
params; the winning params get copied into train.py's build_models() by hand
(with a comment noting they came from this script, and when). This script
itself isn't part of the regular train/serve path -- it's a one-time tuning
step, re-run only when the dataset or approach changes enough to justify it.

Uses the exact same dev/held-out split and StratifiedKFold as train.py, and
never touches the held-out set -- tuning on held-out data would invalidate
the final held-out score reported in train.py.

Run:
    python tune.py
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from train import DEFAULT_DATA_PATH, HOLDOUT_SIZE, N_FOLDS, RANDOM_STATE, load_data

REPORTS_DIR = Path(__file__).parent / "reports"

PARAM_GRIDS = {
    "naive_bayes": {
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", MultinomialNB()),
        ]),
        "grid": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 3],
            "model__alpha": [0.1, 0.5, 1.0],
        },
    },
    "logistic_regression": {
        "pipeline": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "grid": {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2, 3],
            "model__C": [0.1, 1, 10],
        },
    },
}


def main():
    df = load_data(DEFAULT_DATA_PATH)
    x = df["ticket_text"]
    y = df["category"]

    # Same split as train.py -- held-out set is never passed to GridSearchCV below.
    x_dev, _x_holdout, y_dev, _y_holdout = train_test_split(
        x, y, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, spec in PARAM_GRIDS.items():
        print(f"\nGrid searching {name} ({len(spec['grid']['tfidf__ngram_range']) * len(spec['grid']['tfidf__min_df']) * 3} combinations x {N_FOLDS} folds)...")
        search = GridSearchCV(
            spec["pipeline"],
            spec["grid"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(x_dev, y_dev)

        best_idx = search.best_index_
        best_std = search.cv_results_["std_test_score"][best_idx]

        print(f"{name.upper()} best params: {search.best_params_}")
        print(f"{name.upper()} best CV macro-F1: {search.best_score_:.4f} +/- {best_std:.4f}")

        rows.append({
            "model": name,
            "best_params": search.best_params_,
            "best_macro_f1_mean": search.best_score_,
            "best_macro_f1_std": best_std,
        })

    REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "grid_search_results.csv", index=False)
    print(f"\nSaved to {REPORTS_DIR / 'grid_search_results.csv'}")


if __name__ == "__main__":
    main()
