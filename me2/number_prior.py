"""A prior over timer numbers, for the ties the digit heads cannot break.

The timer number is read by two heads, tens and ones, which decide separately
from the same audio. Their commonest mistake is a digit in the wrong place or
in both - "fifty" read as 55, "fifteen" as 55, "thirty" as 33, "one" as 11
(evaluate_numbers.py). Scoring every valid number from both heads together,
plus a weighted log-prior of how often people say it, lets a near tie between
15 and 55 go to the number people actually set timers for.

The prior counts the real timer commands in the training set (STOP, Timers
and Such), add-one smoothed so that no number is impossible. Its weight is
tuned on the validation split by evaluate_numbers.py, which writes both to
vcm_demo/vcm/number_prior.json - the file the demo runtime reads, so the demo
and the evaluation decode the same way.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
PRIOR_FILE = REPO / "vcm_demo/vcm/number_prior.json"
REAL_SOURCES = {"stop", "timers_and_such", "slurp"}
NUMBERS = np.arange(1, 61)


def real_counts(manifest: Path = REPO / "data/dataset/manifest_train.jsonl") -> dict[int, int]:
    """Real timer commands per number in a manifest."""
    counts = {int(n): 0 for n in NUMBERS}
    with open(manifest) as handle:
        for line in handle:
            row = json.loads(line)
            number = row["slots"].get("number") if row["intent"] == "timer.set" else None
            if row.get("model") in REAL_SOURCES and number in counts:
                counts[number] += 1

    return counts


def log_prior(counts: dict, smoothing: float = 1.0) -> np.ndarray:
    """Smoothed log-probability of each number 1-60, (60,), from counts keyed by int or str."""
    c = np.array([counts.get(int(n), counts.get(str(n), 0)) for n in NUMBERS], dtype=np.float64) + smoothing

    return np.log(c / c.sum())


def joint(tens_logp: np.ndarray, ones_logp: np.ndarray, tens_classes: list, ones_classes: list) -> np.ndarray:
    """
    Log-probability of each number 1-60 from both digit heads together.

    Args:
        tens_logp: Tens-head log-probabilities, (B, T)
        ones_logp: Ones-head log-probabilities, (B, O)
        tens_classes: Value of each tens class, "N/A" included
        ones_classes: Value of each ones class, "N/A" included

    Returns:
        Joint log-probabilities, (B, 60), column k for the number k + 1
    """
    return tens_logp[:, [tens_classes.index(n // 10) for n in NUMBERS]] \
        + ones_logp[:, [ones_classes.index(n % 10) for n in NUMBERS]]


def read(scores: np.ndarray, prior: np.ndarray, weight: float) -> np.ndarray:
    """The most likely number per row once the weighted prior is added, (B,)"""
    return NUMBERS[np.argmax(scores + weight * prior, axis=1)]


def load(path: Path = PRIOR_FILE) -> tuple[np.ndarray, float]:
    """The deployed prior, as (log-prior (60,), weight)."""
    spec = json.loads(Path(path).read_text())

    return log_prior(spec["counts"], spec["smoothing"]), float(spec["weight"])
