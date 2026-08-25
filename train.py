"""
train.py

Loads Banking77 (data/banking77.csv, produced by load_data.py), evaluates
Naive Bayes vs Logistic Regression against a most-frequent-class baseline
using stratified 5-fold cross-validation, and picks a winner with an explicit
statistical rule instead of eyeballing a single train/test split (which is
what the old version of this script did, and why its "best model" pick
turned out to be noise).

A 15% held-out set is split off before any of that happens and is only
touched once, at the very end, to report a final number.

Run:
    python load_data.py   # first time only: downloads and caches the dataset
    python train.py                        # trains on data/banking77.csv
    python train.py legacy/support_tickets.csv   # or point it at the old data
"""

import sys
from pathlib import Path

import joblib
import matplotlib  # noqa: E402 -- backend must be set before pyplot import
matplotlib.use("Agg")  # headless: no display available when this runs in CI/CLI
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "banking77.csv"
MODEL_PATH = Path(__file__).parent / "best_ticket_classifier.joblib"
REPORTS_DIR = Path(__file__).parent / "reports"

N_FOLDS = 5
HOLDOUT_SIZE = 0.15
RANDOM_STATE = 42


def load_data(filepath: Path) -> pd.DataFrame:
    """Load ticket dataset from a CSV file."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}. Run `python load_data.py` first "
            "(or pass a path to an existing CSV, e.g. legacy/support_tickets.csv)."
        )

    try:
        df = pd.read_csv(filepath)
    except EmptyDataError as exc:
        raise ValueError(
            f"Dataset is empty: {filepath}. Add rows with "
            "'ticket_text' and 'category' columns."
        ) from exc

    required_columns = {"ticket_text", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain these columns: {required_columns}")

    df = df.dropna(subset=["ticket_text", "category"])
    df["ticket_text"] = df["ticket_text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["ticket_text"] != "") & (df["category"] != "")]

    if df.empty:
        raise ValueError(
            "Dataset has no usable rows after cleanup. Ensure ticket_text "
            "and category contain non-empty values."
        )

    return df


def build_models() -> dict:
    """Create ML pipelines for comparison. Hyperparameters below came from
    GridSearchCV (see tune.py), run 2026-08-03 on data/banking77.csv's dev
    split -- ngram_range in {(1,1),(1,2)}, min_df in {1,2,3}, alpha/C in
    {0.1,0.5,1.0}/{0.1,1,10}. Re-run tune.py if the dataset changes enough
    to make these stale."""
    models = {
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("model", MultinomialNB(alpha=0.1))
        ]),
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1)),
            ("model", LogisticRegression(max_iter=1000, C=10))
        ])
    }
    return models


def run_cross_validation(models: dict, x_dev, y_dev, cv) -> dict:
    """5-fold stratified CV for the baseline and every candidate model.
    Returns {name: scores_array} so mean/std can be compared afterward."""
    results = {}

    baseline = DummyClassifier(strategy="most_frequent")
    baseline_scores = cross_val_score(
        baseline, x_dev, y_dev, cv=cv, scoring="f1_macro"
    )
    results["baseline_most_frequent"] = baseline_scores
    print(
        f"BASELINE (most_frequent)   macro-F1: {baseline_scores.mean():.4f} "
        f"+/- {baseline_scores.std():.4f}"
    )

    for name, pipeline in models.items():
        scores = cross_val_score(
            pipeline, x_dev, y_dev, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        results[name] = scores
        print(
            f"{name.upper():<25} macro-F1: {scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    return results


def select_model(models: dict, results: dict) -> str:
    """Pick a winner only if the CV means differ by more than one standard
    deviation. Otherwise the gap is noise -- default to the simpler, faster
    model and say so explicitly."""
    names = list(models.keys())
    name_a, name_b = names[0], names[1]
    mean_a, std_a = results[name_a].mean(), results[name_a].std()
    mean_b, std_b = results[name_b].mean(), results[name_b].std()

    diff = abs(mean_a - mean_b)
    threshold = max(std_a, std_b)
    higher_name = name_a if mean_a > mean_b else name_b

    print("\nMODEL SELECTION")
    print(f"Mean macro-F1 difference: {diff:.4f}")
    print(f"Threshold (larger of the two CV standard deviations): {threshold:.4f}")

    if diff > threshold:
        print(
            f"{higher_name} beats the other model by more than one standard "
            f"deviation -> selecting {higher_name}."
        )
        return higher_name

    print(
        "Difference is within one standard deviation -- not statistically "
        "distinguishable from noise. Defaulting to naive_bayes: it has no "
        "hyperparameters to tune, trains in one pass over the data (no "
        "iterative solver like logistic regression), and predicts faster."
    )
    return "naive_bayes"


def save_confusion_matrix(y_true, y_pred, labels, chosen_name: str) -> None:
    """Save a confusion matrix heatmap plus a CSV of the 10 most-confused
    class pairs, built from cross-validated (not held-out) predictions."""
    REPORTS_DIR.mkdir(exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.22), max(10, len(labels) * 0.22)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({chosen_name}, 5-fold CV predictions)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append((true_label, pred_label, int(cm[i, j])))
    pairs.sort(key=lambda row: row[2], reverse=True)

    top_pairs = pd.DataFrame(
        pairs[:10], columns=["true_label", "predicted_label", "count"]
    )
    top_pairs.to_csv(REPORTS_DIR / "confused_pairs.csv", index=False)

    print(f"\nSaved confusion matrix to {REPORTS_DIR / 'confusion_matrix.png'}")
    print(f"Saved top confused pairs to {REPORTS_DIR / 'confused_pairs.csv'}")
    print("\nTop 5 most-confused class pairs:")
    for true_label, pred_label, count in pairs[:5]:
        print(f"  {true_label} -> {pred_label}: {count}")


def save_model(model, filepath: Path) -> None:
    joblib.dump(model, filepath)
    print(f"\nSaved chosen model to: {filepath}")


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows, {df['category'].nunique()} classes from {data_path}\n")

    x = df["ticket_text"]
    y = df["category"]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 categories to train a classifier.")

    # Held out once, up front. Nothing below this line touches x_holdout /
    # y_holdout until the single evaluation call at the very end.
    x_dev, x_holdout, y_dev, y_holdout = train_test_split(
        x, y, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = build_models()
    results = run_cross_validation(models, x_dev, y_dev, cv)

    chosen_name = select_model(models, results)
    chosen_pipeline = models[chosen_name]

    y_pred_cv = cross_val_predict(chosen_pipeline, x_dev, y_dev, cv=cv, n_jobs=-1)
    labels = sorted(y_dev.unique())
    save_confusion_matrix(y_dev, y_pred_cv, labels, chosen_name)

    chosen_pipeline.fit(x_dev, y_dev)
    y_pred_holdout = chosen_pipeline.predict(x_holdout)
    holdout_f1 = f1_score(y_holdout, y_pred_holdout, average="macro")

    print("\nFINAL HELD-OUT RESULT (touched once)")
    print(f"Model: {chosen_name}")
    print(f"Held-out macro-F1 ({len(x_holdout)} samples): {holdout_f1:.4f}")

    save_model(chosen_pipeline, MODEL_PATH)


if __name__ == "__main__":
    main()
