import json
from pathlib import Path

from src.evaluation.metrics import score
from src.evaluation.parsing import parse_labels
from src.models.base import Model
from src.utils.files import write_jsonl

SCORES_DIR = "scores"


def scores_stem(split: str, max_samples: int | None) -> str:
    """Get a file name stem for the scores of one partition/

    Args:
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.

    Returns:
        str: A file name stem.
    """
    if max_samples is None:
        return split

    return f"{split}.sample{max_samples}"


def score_examples(
    model: Model,
    examples: list[dict],
    conversations: list[list[dict]],
    lang: str,
    predictions_path: Path,
) -> dict[str, float]:
    """Score one partition of one fine-tuned run by letting the model answer every example and
    comparing its output to the gold answer.

    Args:
        model (Model): The loaded model.
        examples (list[dict]): The conversational examples, whose last message is the gold
            answer and whose second-to-last is the subtitle.
        conversations (list[list[dict]]): The prompt the model answers for each example.
        lang (str): The language of the examples.
        predictions_path (Path): The JSON Lines file for the per-line predictions.

    Returns:
        dict[str, float]: A dictionary of the scores.
    """
    answers = model.generate(conversations)
    parsed = [parse_labels(answer) for answer in answers]
    gold = [json.loads(example["messages"][-1]["content"]) for example in examples]

    write_jsonl(
        predictions_path,
        [
            {
                "align_id": example["align_id"],
                "lang": lang,
                "subtitle": example["messages"][-2]["content"],
                "gold": labels,
                "output": answer,
                "predicted": output.labels,
                "malformed": output.malformed,
            }
            for example, labels, answer, output in zip(examples, gold, answers, parsed, strict=True)
        ],
    )

    return score(gold, parsed)
