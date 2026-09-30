import json
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, jaccard_score

from src.constants import EMOTIONS

# Documented zero-division policy (§7): a metric that divides by zero counts as 0.
ZERO_DIVISION = 0.0


@dataclass
class ParsedOutput:
    """The permitted labels found in one model answer, and whether the answer was malformed."""

    labels: list[str]
    malformed: bool


def parse_labels(output: str) -> ParsedOutput:
    """Parse a model answer into permitted emotion labels, following the §3 parsing rules.

    Temporary parser for dev-set selection (#12); to be replaced by the team's parser (#28).

    Args:
        output (str): The raw model answer.

    Returns:
        ParsedOutput: The permitted labels in fixed order, and whether the answer was malformed.
    """
    try:
        value = json.loads(output.strip())

    except json.JSONDecodeError:
        return ParsedOutput(labels=[], malformed=True)

    if not isinstance(value, list):
        return ParsedOutput(labels=[], malformed=True)

    found = {" ".join(item.split()).lower() for item in value if isinstance(item, str)}

    return ParsedOutput(
        labels=[emotion for emotion in EMOTIONS if emotion in found], malformed=False
    )


def binarise(label_sets: list[list[str]]) -> np.ndarray:
    """Turn label sets into 0/1 vectors in the fixed EMOTIONS order.

    Args:
        label_sets (list[list[str]]): One list of labels per line.

    Returns:
        np.ndarray: A matrix with one row per line and one column per emotion.
    """
    return np.array([[int(emotion in labels) for emotion in EMOTIONS] for labels in label_sets])


def score(gold: list[list[str]], predicted: list[ParsedOutput]) -> dict[str, float]:
    """Compute the primary §7 metrics and the malformed-output rate.

    Args:
        gold (list[list[str]]): The true labels of each line.
        predicted (list[ParsedOutput]): The parsed model answer for each line.

    Returns:
        dict[str, float]: The metrics.
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
