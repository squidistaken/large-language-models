import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.constants import EVAL_SPLITS
from src.data.loading import load_examples
from src.evaluation.metrics import metrics_info, score
from src.evaluation.parsing import parse_labels
from src.models.base import Model
from src.utils import tracking
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


class BaseScorer(ABC):
    """Class that scores a model on one partition of one condition and saves the scores and predictions."""

    def __init__(
        self,
        model: Model,
        run_dir: Path,
        split: str,
        max_samples: int | None,
        wandb_cfg: DictConfig,
        logger: Logger,
    ) -> None:
        """Initialise the class.

        Args:
            model (Model): The model.
            run_dir (Path): The run folder to save the scores and predictions in.
            split (str): The partition name.
            max_samples (int | None): The number of lines scored per language, or None for all.
            wandb_cfg (DictConfig): The wandb config.
            logger (Logger): The logger.

        Raises:
            ValueError: If the split is not dev or test.
        """
        check_eval_split(split)

        self.model = model
        self.run_dir = run_dir
        self.split = split
        self.max_samples = max_samples
        self.wandb_cfg = wandb_cfg
        self.logger = logger

    @property
    @abstractmethod
    def condition(self) -> str:
        """Get the name of the scored condition.

        Returns:
            str: A condition.
        """

    @property
    @abstractmethod
    def langs(self) -> list[str]:
        """Get the languages to score.

        Returns:
            list[str]: A list of language codes.
        """

    @property
    @abstractmethod
    def data(self) -> tuple[str, str, int]:
        """Get where the scored lines come from.

        Returns:
            tuple[str, str, int]: The processed-splits folder, the target language, and the seed
                                  of the random subset.
        """

    @abstractmethod
    def _tracking_run(self, model_info: dict[str, Any]) -> Any:
        """Start or resume the W&B run the scores are logged to.

        Args:
            model_info (dict[str, Any]): The model's info.

        Returns:
            Any: A W&B run, or None if W&B is switched off.
        """

    def _attach(self) -> None:
        """Prepare the loaded model before it answers, e.g. by attaching an adapter. Nothing by
        default."""
        return None

    def _save(self, model_info: dict[str, Any]) -> None:
        """Save anything else about the run after it has been scored. Nothing by default.

        Args:
            model_info (dict[str, Any]): The model's info.
        """
        return None

    def _prepare(self, examples: list[dict], lang: str) -> list[dict]:
        """Change the examples before the model answers them, e.g. by adding demonstrations.
        Unchanged by default.

        Args:
            examples (list[dict]): The conversational examples.
            lang (str): The language of the examples.

        Returns:
            list[dict]: A list of the examples to score.
        """
        return examples

    def score(self) -> ScoreSummary:
        """Let the model answer the partition in every language, then save and log the scores.

        Returns:
            ScoreSummary: A summary of the scores and predictions of the partition.
        """
        self.model.load()

        try:
            self._attach()
            summary = self._score_languages()
            model_info = self.model.info()

        finally:
            self.model.unload()

        self._save(model_info)

        wandb_run = self._tracking_run(model_info)

        tracking.log_scores(wandb_run, summary.split, summary.scores, summary.predictions)
        tracking.finish(wandb_run)

        return summary

    def _score_languages(self) -> ScoreSummary:
        """Score the loaded model on the partition of every language and save the scores.

        Returns:
            ScoreSummary: A summary of the scores and predictions of the partition.
        """
        data_dir, target_lang, seed = self.data
        summary = ScoreSummary(
            condition=self.condition,
            split=self.split,
            scores_file=str(scores_path(self.run_dir, self.split, self.max_samples)),
        )

        for lang in self.langs:
            examples = load_examples(
                data_dir, target_lang, [lang], self.split, self.logger, self.max_samples, seed
            )
            examples = self._prepare(examples, lang)
            summary.scores[lang], summary.predictions[lang] = score_examples(
                self.model,
                examples,
                lang,
                predictions_path(self.run_dir, self.split, self.max_samples, lang),
            )

            self.logger.info(f"{lang} {self.split}: {summary.scores[lang]}")

        write_json(
            Path(summary.scores_file),
            {
                "split": self.split,
                "max_samples": self.max_samples,
                "model": self.model.info(),
                "metrics": metrics_info(),
                "scores": summary.scores,
            },
        )

        return summary
