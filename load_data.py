"""
load_data.py

Fetches the Banking77 dataset and caches it locally as a single CSV with the
same column names the rest of this project already uses (ticket_text,
category), so train.py doesn't need to know where the data came from.

Banking77 is 13,083 real customer service queries about banking, labeled with
77 fine-grained intents (card issues, transfers, exchange rates, etc). It
replaces the old synthetic support_tickets.csv (see legacy/).

Why we don't use `datasets.load_dataset("PolyAI/banking77")`: Hugging Face
deprecated script-based dataset loading, and PolyAI/banking77 is still a
script-based dataset. Trying to load it with a current version of the
`datasets` library fails ("Dataset scripts are no longer supported"). The
script itself just downloads two plain CSVs from PolyAI's GitHub repo, so we
fetch those directly with httpx (already a project dependency) instead of
pulling in `datasets` + `pyarrow` (~57MB of new dependencies) for two CSV
files.

Run:
    python load_data.py
"""

import sys
from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "banking77.csv"

TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"


def fetch_csv(url: str) -> pd.DataFrame:
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(response.text))


def download_banking77() -> pd.DataFrame:
    """Download and combine Banking77's train + test CSVs into one frame."""
    train_df = fetch_csv(TRAIN_URL)
    test_df = fetch_csv(TEST_URL)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.rename(columns={"text": "ticket_text"})
    return df[["ticket_text", "category"]]


def print_stats(df: pd.DataFrame) -> None:
    class_counts = df["category"].value_counts()
    duplicate_rate = df.duplicated(subset=["ticket_text"]).mean()

    print(f"Row count: {len(df)}")
    print(f"Class count: {df['category'].nunique()}")
    print(f"Min class size: {class_counts.min()} ({class_counts.idxmin()})")
    print(f"Max class size: {class_counts.max()} ({class_counts.idxmax()})")
    print(f"Duplicate ticket_text rate: {duplicate_rate:.4f}")
    print("\nClass distribution:")
    for category, count in class_counts.items():
        print(f"  {category}: {count}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Using cached dataset at {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        print("Downloading Banking77 from PolyAI's GitHub repo...")
        try:
            df = download_banking77()
        except httpx.HTTPError as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            sys.exit(1)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")

    print()
    print_stats(df)


if __name__ == "__main__":
    main()
