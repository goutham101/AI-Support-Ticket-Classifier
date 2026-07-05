import os
import sys
import joblib


def load_model(filepath: str):
    """Load a saved model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file not found: {filepath}. Run train.py first."
        )
    return joblib.load(filepath)


def main():
    model_path = "best_ticket_classifier.joblib"
    model = load_model(model_path)

    if len(sys.argv) > 1:
        ticket_text = " ".join(sys.argv[1:])
    else:
        ticket_text = input("Enter a support ticket: ").strip()

    if not ticket_text:
        print("No ticket text provided.")
        return

    prediction = model.predict([ticket_text])[0]
    print(f"Predicted category: {prediction}")


if __name__ == "__main__":
    main()