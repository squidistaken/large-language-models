import numpy as np
from sklearn.metrics import f1_score, jaccard_score

from src.constants import EMOTIONS
from src.evaluation.parsing import ParsedOutput

# A metric that divides by zero counts as 0.
ZERO_DIVISION = 0.0


def binarise(label_sets: list[list[str]]) -> np.ndarray:
    """Turn label sets into 0/1 vectors in the fixed EMOTIONS order.

    Args:
        label_sets (list[list[str]]): The true or predicted labels of each line.

    Returns:
        np.ndarray: A matrix with one row per line and one column per emotion.
    """
    return np.array([[int(emotion in labels) for emotion in EMOTIONS] for labels in label_sets])


# TODO: Compute all metrics required for the project.
def score(gold: list[list[str]], predicted: list[ParsedOutput]) -> dict[str, float]:
    """Score a set of predictions against the gold labels.

    Args:
        gold (list[list[str]]): The true labels of each line.
        predicted (list[ParsedOutput]): The parsed model answer for each line.

    Returns:
        dict[str, float]: A dictionary with the metrics.
    """
    y_true = binarise(gold)
    y_pred = binarise([output.labels for output in predicted])
    count = len(predicted)

    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=ZERO_DIVISION)),
        "jaccard_samples": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=ZERO_DIVISION)
        ),
        "malformed_rate": sum(output.malformed for output in predicted) / count,
        "empty_rate": sum(not output.labels for output in predicted) / count,
        "count": count,
    }
