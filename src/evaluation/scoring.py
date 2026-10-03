import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.constants import EVAL_SPLITS
from src.data.loading import load_examples
from src.evaluation.metrics import metrics_info, score
from src.evaluation.parsing import parse_labels
from src.models.base import Model
from src.utils.files import write_json, write_jsonl
from src.utils.logger import Logger
from src.utils.runs import predictions_path, scores_path


@dataclass
class ScoreSummary:
    """The scores and predictions of one partition of one condition."""

    condition: str
    split: str
    scores_file: str
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    predictions: dict[str, list[dict]] = field(default_factory=dict)


def check_eval_split(split: str) -> None:
    """Check that the split is one of the permitted evaluation splits.

    Args:
        split (str): The partition name.

    Raises:
        ValueError: If the split is not dev or test.
    """
    if split not in EVAL_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Choose one of: {', '.join(EVAL_SPLITS)}")


def score_examples(
    model: Model,
    examples: list[dict],
    lang: str,
    path: Path | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """Score one partition by letting the model answer every example and comparing its output to
    the gold answer.

    Args:
        model (Model): The loaded model.
        examples (list[dict]): The conversational examples, whose last message is the gold
                               answer and whose second-to-last is the subtitle.
        lang (str): The language of the examples.
        path (Path | None): The JSON Lines file for the per-line predictions, or None to keep
                            them in memory only.

    Returns:
        tuple[dict[str, Any], list[dict]]: The scores and the per-line predictions.
    """
    answers = model.generate([example["messages"][:-1] for example in examples])
    parsed = [parse_labels(answer) for answer in answers]
    gold = [json.loads(example["messages"][-1]["content"]) for example in examples]
    rows = [
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
    ]

    if path is not None:
        write_jsonl(path, rows)

    return score(gold, parsed), rows


def score_languages(
    model: Model,
    condition: str,
    data_dir: str,
    target_lang: str,
    langs: list[str],
    split: str,
    max_samples: int | None,
    seed: int,
    run: Path,
    logger: Logger,
) -> ScoreSummary:
    """Score a loaded model on one partition of every given language and save the scores.

    Args:
        model (Model): The loaded model, with any adapter already attached.
        condition (str): The name of the condition.
        data_dir (str): The base directory of the processed splits.
        target_lang (str): The assigned target language.
        langs (list[str]): The languages to score.
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.
        seed (int): The seed of the random subset.
        run (Path): The run folder to save the scores and predictions in.
        logger (Logger): The logger.

    Returns:
        ScoreSummary: A summary of the scores and predictions of the partition.
    """
    summary = ScoreSummary(
        condition=condition,
        split=split,
        scores_file=str(scores_path(run, split, max_samples)),
    )

    for lang in langs:
        examples = load_examples(data_dir, target_lang, [lang], split, logger, max_samples, seed)
        summary.scores[lang], summary.predictions[lang] = score_examples(
            model, examples, lang, predictions_path(run, split, max_samples, lang)
        )

        logger.info(f"{lang} {split}: {summary.scores[lang]}")

    write_json(
        Path(summary.scores_file),
        {
            "split": split,
            "max_samples": max_samples,
            "model": model.info(),
            "metrics": metrics_info(),
            "scores": summary.scores,
        },
    )

    return summary
