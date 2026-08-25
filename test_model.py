"""
test_model.py

Model-quality tests, separate from test_main.py's API-contract tests. The
regression test actually trains models (5-fold CV), so it's marked @slow and
excluded from the fast suite -- see pytest.ini.

Run:
    pytest                    # everything, including the slow regression test
    pytest -m "not slow"      # fast suite only
"""

from pathlib import Path

import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from train import build_models, load_data

SAMPLE_DATA_PATH = Path(__file__).parent / "sample_tickets.csv"

# Measured on sample_tickets.csv (40 rows, 8 classes, 5 examples each) on
# 2026-08-04, running the tuned logistic_regression pipeline from
# build_models() through the same 5-fold StratifiedKFold CV protocol as
# train.py: mean macro-F1 = 0.6000 (fold scores: 0.833, 0.562, 0.500, 0.396,
# 0.708 -- high variance is expected with only ~8 examples per fold on a
# dataset this small). Floor is 5 points below that measured mean. This is a
# coarse "did something obviously break" guard, not a precision benchmark --
# see train.py's own output for the real number on the full 13,083-row
# Banking77 dataset.
REGRESSION_FLOOR = 0.55


@pytest.mark.slow
def test_cv_macro_f1_regression():
    df = load_data(SAMPLE_DATA_PATH)
    x = df["ticket_text"]
    y = df["category"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = build_models()["logistic_regression"]
    scores = cross_val_score(pipeline, x, y, cv=cv, scoring="f1_macro")

    assert scores.mean() > REGRESSION_FLOOR, (
        f"CV macro-F1 dropped to {scores.mean():.4f} on sample_tickets.csv, "
        f"below the {REGRESSION_FLOOR} floor. Did build_models() change?"
    )
