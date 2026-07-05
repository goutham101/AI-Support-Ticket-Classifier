import os
import joblib
import pandas as pd

from pandas.errors import EmptyDataError
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath: str) -> pd.DataFrame:
    """Load ticket dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except EmptyDataError as exc:
        raise ValueError(
            f"Dataset is empty: {filepath}. Add rows with "
            "'ticket_text' and 'category' columns."
        ) from exc

    required_columns = {"ticket_text", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {required_columns}"
        )

    df = df.dropna(subset=["ticket_text", "category"])
    df["ticket_text"] = df["ticket_text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[
        (df["ticket_text"] != "")
        & (df["category"] != "")
    ]

    if df.empty:
        raise ValueError(
            "Dataset has no usable rows after cleanup. Ensure ticket_text "
            "and category contain non-empty values."
        )

    return df


def build_models() -> dict:
    """Create ML pipelines for comparison."""
    models = {
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("model", MultinomialNB())
        ]),
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=1000, C=10))
        ])
    }
    return models


def evaluate_models(models: dict, x_train, x_test, y_train, y_test) -> tuple:
    """Train and evaluate models. Return best model info."""
    best_model_name = None
    best_model = None
    best_accuracy = -1

    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{name.upper()}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions, zero_division=0))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = pipeline

    return best_model_name, best_model, best_accuracy


def save_model(model, filepath: str) -> None:
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"\nSaved best model to: {filepath}")


def main():
    data_path = "support_tickets.csv"
    model_path = "best_ticket_classifier.joblib"

    df = load_data(data_path)

    x = df["ticket_text"]
    y = df["category"]

    if y.nunique() < 2:
        raise ValueError(
            "Need at least 2 categories to train a classifier."
        )

    num_classes = y.nunique()
    num_rows = len(df)
    min_class_count = y.value_counts().min()
    stratify = y if min_class_count >= 2 else None
    test_size = 0.25

    # Stratified split needs at least one test sample per class.
    if stratify is not None and int(num_rows * test_size) < num_classes:
        if num_classes >= num_rows:
            raise ValueError(
                "Not enough rows to split data with all categories represented."
            )
        test_size = num_classes

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify
    )

    models = build_models()
    best_model_name, best_model, best_accuracy = evaluate_models(
        models, x_train, x_test, y_train, y_test
    )

    print("\nBEST MODEL")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")

    save_model(best_model, model_path)


if __name__ == "__main__":
    main()
