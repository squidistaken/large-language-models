from importlib.metadata import version
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.constants import EMOTIONS
from src.evaluation.parsing import ParsedOutput

# A metric that divides by zero counts as 0.
ZERO_DIVISION = 0.0

PRIMARY_METRICS = ("micro_f1", "jaccard_samples")
SECONDARY_METRICS = ("macro_f1", "micro_precision", "micro_recall", "hamming_loss")
DIAGNOSTIC_METRICS = ("exact_match", "malformed_rate", "empty_rate")
AGGREGATE_METRICS = (*PRIMARY_METRICS, *SECONDARY_METRICS, *DIAGNOSTIC_METRICS)

SHORT_NAMES = {
    "micro_f1": "micro-F1",
    "jaccard_samples": "Jaccard",
    "macro_f1": "macro-F1",
    "micro_precision": "micro-P",
    "micro_recall": "micro-R",
    "hamming_loss": "Hamming",
    "exact_match": "Exact",
    "malformed_rate": "Malformed",
    "empty_rate": "Empty",
}

LOWER_IS_BETTER = {"hamming_loss", "malformed_rate", "empty_rate"}


def metric_label(key: str, short: bool = True) -> str:
    """Get a label for a metric, optionally using the short name and indicating whether
    lower is better.

    Args:
        key (str): The metric key.
        short (bool): Whether to use the short name instead of the key.

    Returns:
        str: A label.
    """
    return f"{SHORT_NAMES[key] if short else key}{' ↓' if key in LOWER_IS_BETTER else ''}"


def binarise(label_sets: list[list[str]]) -> np.ndarray:
    """Turn label sets into 0/1 vectors in the fixed EMOTIONS order.

    Args:
        label_sets (list[list[str]]): The true or predicted labels of each line.

    Returns:
        np.ndarray: A matrix with one row per line and one column per emotion.
    """
    return np.array(
        [[int(emotion in labels) for emotion in EMOTIONS] for labels in label_sets]
    ).reshape(-1, len(EMOTIONS))


def metrics_info() -> dict[str, Any]:
    """Describe how the metrics are computed, for the scores file.

    Returns:
        dict[str, Any]: A dictionary with the label order, zero-division policy and library.
    """
    return {
        "label_order": list(EMOTIONS),
        "zero_division": ZERO_DIVISION,
        "zero_division_policy": (
            f"A metric whose denominator is zero (no positive gold labels or no predictions for a "
            f"class or line) is set to {ZERO_DIVISION}."
        ),
        "malformed_policy": (
            "Malformed answers and answers with no permitted label count as an empty predicted "
            "label set."
        ),
        "implementation": f"scikit-learn {version('scikit-learn')}",
    }


def score(gold: list[list[str]], predicted: list[ParsedOutput]) -> dict[str, Any]:
    """Score a set of predictions against the gold labels with every metric of the brief.

    Args:
        gold (list[list[str]]): The true labels of each line.
        predicted (list[ParsedOutput]): The parsed model answer for each line.

    Returns:
        dict[str, Any]: A dictionary with the metrics and their values,
                        plus the count of lines.
    """
    y_true = binarise(gold)
    y_pred = binarise([output.labels for output in predicted])
    count = len(predicted)
    zero = {"zero_division": ZERO_DIVISION}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(EMOTIONS))), **zero
    )

    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", **zero)),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", **zero)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **zero)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", **zero)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", **zero)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "malformed_rate": sum(output.malformed for output in predicted) / count,
        "empty_rate": sum(not output.labels for output in predicted) / count,
        "count": count,
        "per_emotion": {
            emotion: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, emotion in enumerate(EMOTIONS)
        },
    }


def example_jaccard(gold: list[str], predicted: list[str]) -> float:
    """Get the Jaccard index for a single example, which is the size of the
    intersection divided by the size of the union.

    Args:
        gold (list[str]): The true labels.
        predicted (list[str]): The predicted labels.

    Returns:
        float: A Jaccard index.
    """
    union = set(gold) | set(predicted)

    return len(set(gold) & set(predicted)) / len(union) if union else ZERO_DIVISION


INTERVAL_METRICS = ("micro_f1", "jaccard_samples", "macro_f1")
BOOTSTRAP_SAMPLES = 1000


def bootstrap_intervals(
    gold: list[list[str]],
    predicted: list[list[str]],
    seed: int,
    level: float = 0.95,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, list[float]]:
    """Estimate percentile bootstrap confidence intervals by resampling the lines.

    Args:
        gold (list[list[str]]): The true labels of each line.
        predicted (list[list[str]]): The predicted labels of each line.
        seed (int): The seed of the resampling.
        level (float): The confidence level.
        samples (int): The number of bootstrap samples.

    Returns:
        dict[str, list[float]]: A dictionary with the metrics and their confidence intervals.
    """
    y_true, y_pred = binarise(gold).astype(bool), binarise(predicted).astype(bool)
    true_pos, false_pos, false_neg = y_true & y_pred, ~y_true & y_pred, y_true & ~y_pred
    union = (y_true | y_pred).sum(axis=1)
    line_jaccard = np.divide(
        true_pos.sum(axis=1), union, out=np.full(len(union), ZERO_DIVISION), where=union > 0
    )
    resamples = np.random.default_rng(seed).integers(0, len(gold), size=(samples, len(gold)))
    tp, fp, fn = (counts[resamples].sum(axis=1) for counts in (true_pos, false_pos, false_neg))

    def f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        denominator = 2 * tp + fp + fn

        return np.divide(
            2 * tp, denominator, out=np.full(tp.shape, ZERO_DIVISION), where=denominator > 0
        )

    estimates = {
        "micro_f1": f1(tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1)),
        "jaccard_samples": line_jaccard[resamples].mean(axis=1),
        "macro_f1": f1(tp, fp, fn).mean(axis=1),
    }
    tail = (1 - level) / 2 * 100

    return {
        metric: [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]
        for metric, values in estimates.items()
    }
