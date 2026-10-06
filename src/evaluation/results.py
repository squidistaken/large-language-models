from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import AGGREGATE_METRICS, RUN_INFO_FILE, TRAIN, ZERO_SHOT
from src.data.writing import SplitWriter
from src.evaluation.metrics import bootstrap_intervals, metrics_info, per_emotion_rows
from src.evaluation.scoring import ScoreSummary
from src.finetune.conditions import (
    condition_languages,
    conditions,
    evaluation_languages,
    system_name,
)
from src.finetune.evaluate import AdapterScorer
from src.finetune.selection import ConfigSelector, SelectionSummary
from src.prompting.scorer import PromptScorer
from src.utils import tracking
from src.utils.files import read_json, read_jsonl, write_json
from src.utils.logger import Logger
from src.utils.runs import (
    RunVariant,
    pinned_model_cfg,
    predictions_path,
    read_run_info,
    results_path,
    run_dir,
    scores_path,
)

Scores = dict[str, dict[str, Any]]
Variants = dict[str, list[tuple[Path, RunVariant]]]


@dataclass
class ResultsSummary:
    """The final results table of one partition."""

    run_name: str
    split: str
    max_samples: int | None
    rows: list[dict[str, Any]] = field(default_factory=list)
    per_emotion: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[dict[str, Any]] = field(default_factory=list)
    curve: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Get the results as they are saved.

        Returns:
            dict[str, Any]: A mapping with the settings, the metric definitions, and every table.
        """
        return {
            "run_name": self.run_name,
            "split": self.split,
            "max_samples": self.max_samples,
            "metrics": metrics_info(),
            "results": self.rows,
            "per_emotion": self.per_emotion,
            "seeds": self.seeds,
            "learning_curve": self.curve,
        }

    @classmethod
    def load(cls, path: Path) -> "ResultsSummary":
        """Load saved results.

        Args:
            path (Path): The results file.

        Returns:
            ResultsSummary: A summary of the results.
        """
        saved = read_json(path)

        return cls(
            saved["run_name"],
            saved["split"],
            saved["max_samples"],
            saved["results"],
            saved["per_emotion"],
            saved.get("seeds", []),
            saved.get("learning_curve", []),
            [str(path)],
        )


class FinalEvaluator:
    """Class that scores every system on a partition and writes the results table."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The full config.
            logger (Logger): The logger.
        """
        self.cfg = cfg
        self.logger = logger
        self.split = str(cfg.evaluate.split)
        self.max_samples = cfg.evaluate.max_samples
        self.selection: SelectionSummary | None = None

        if cfg.evaluate.run_name:
            self.run_name = str(cfg.evaluate.run_name)

        else:
            self.selection = ConfigSelector(cfg.finetune, logger).select()
            self.run_name = self.selection.best.run_name

    def evaluate(self) -> ResultsSummary:
        """Score every missing system on the partition and write the results table.

        Returns:
            ResultsSummary: A summary of the results.
        """
        runs = {
            condition: run_dir(self.cfg.finetune.out_dir, condition, self.run_name)
            for condition in conditions(self.cfg.finetune.target_lang)
        }
        folders: dict[str, Path] = {}
        scores = {}

        if self.cfg.evaluate.include_zero_shot:
            scorer = self._zero_shot_scorer(next(iter(runs.values())))
            folders[ZERO_SHOT] = scorer.run_dir
            scores[ZERO_SHOT] = self._ensure(scorer.run_dir, scorer.score)

        for condition, run in runs.items():
            folders[condition] = run
            scores[condition] = self._adapter_scores(run)

        summary = ResultsSummary(self.run_name, self.split, self.max_samples)

        for condition, by_lang in scores.items():
            for lang, metrics in by_lang.items():
                summary.rows.append(
                    self._row(system_name(condition), folders[condition], lang, metrics)
                )
                summary.per_emotion.extend(
                    {"system": system_name(condition), "eval_lang": lang, **row}
                    for row in per_emotion_rows(metrics)
                )

        repetitions, capped = self._variants()
        summary.seeds = self._seeds(runs, scores, repetitions)
        summary.curve = self._curve(capped, summary.rows)

        path = results_path(self.cfg.evaluate.out_dir, self.split, self.max_samples, self.run_name)
        summary.files = [str(path)]

        write_json(path, summary.to_json())

        self._log(summary)
        self.logger.info(f"Wrote the {self.split} results to {', '.join(summary.files)}")

        return summary

    def _row(self, system: str, folder: Path, lang: str, metrics: dict[str, Any]) -> dict:
        """Get one row of the results table for one system and evaluation language.

        Args:
            system (str): The system name.
            folder (Path): The run folder holding the predictions.
            lang (str): The evaluation language.
            metrics (dict[str, Any]): The scores in that language.

        Returns:
            dict: A row.
        """
        lines = read_jsonl(predictions_path(folder, self.split, self.max_samples, lang))
        intervals = bootstrap_intervals(
            [line["gold"] for line in lines],
            [line["predicted"] for line in lines],
            seed=self.cfg.seed,
        )

        return {
            "system": system,
            "eval_lang": lang,
            **{key: metrics[key] for key in AGGREGATE_METRICS},
            **{f"{key}_ci": bounds for key, bounds in intervals.items()},
            "count": metrics["count"],
        }

    def _variants(self) -> tuple[Variants, Variants]:
        """Get the repetitions and capped runs of the selected configuration.

        Returns:
            tuple[Variants, Variants]: A dictionary of repetitions and a dictionary of capped runs,
                                       keyed by condition, each run with its variant.
        """
        repetitions: Variants = {}
        capped: Variants = {}

        for condition in conditions(self.cfg.finetune.target_lang):
            for info in sorted(
                Path(self.cfg.finetune.out_dir, condition).glob(f"*/{RUN_INFO_FILE}")
            ):
                config = read_run_info(info.parent)["config"]
                kind = RunVariant.of(config)

                if config["run_name"] != self.run_name:
                    continue

                # A run with both another seed and a cap fits neither experiment.
                if kind.is_repetition:
                    repetitions.setdefault(condition, []).append((info.parent, kind))
                elif kind.is_capped:
                    capped.setdefault(condition, []).append((info.parent, kind))

        return repetitions, capped

    def _seeds(
        self, runs: dict[str, Path], scores: dict[str, Scores], repetitions: Variants
    ) -> list[dict[str, Any]]:
        """Get the mean, standard deviation and values of every metric for every system and
        language, across the main run and its repetitions.

        Args:
            runs (dict[str, Path]): The main run of every condition.
            scores (dict[str, Scores]): The scores of the main run of every condition.
            repetitions (Variants): The repetitions per condition.

        Returns:
            list[dict[str, Any]]: A list of results rows, one per system and evaluation language,
                                  with the seeds and the mean, standard deviation and values of
                                  every metric.
        """
        rows = []

        for condition, others in repetitions.items():
            main_seed = read_run_info(runs[condition])["config"]["seed"]
            seeded = sorted(
                [
                    (main_seed, scores[condition]),
                    *((kind.seed, self._adapter_scores(run)) for run, kind in others),
                ],
                key=lambda pair: pair[0],
            )

            for lang in evaluation_languages(condition, self.cfg.finetune.target_lang):
                row: dict[str, Any] = {
                    "system": system_name(condition),
                    "eval_lang": lang,
                    "seeds": [seed for seed, _ in seeded],
                }

                for metric in AGGREGATE_METRICS:
                    values = [scores[lang][metric] for _, scores in seeded]
                    row[metric] = {
                        "mean": mean(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                        "values": values,
                    }

                rows.append(row)

        return rows

    def _curve(self, capped: Variants, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get the learning curve of the selected configuration, across the main run and its capped
        runs.

        Args:
            capped (Variants): The capped runs per condition.
            rows (list[dict[str, Any]]): The results rows of the main runs and zero-shot.

        Returns:
            list[dict[str, Any]]: A list of results rows, one per system and evaluation language,
                                  with the number of lines used to train the system.
        """
        target = str(self.cfg.finetune.target_lang)
        trained = {
            condition: others
            for condition, others in capped.items()
            if target in condition_languages(condition, target)
        }

        if not trained:
            return []

        main = {row["system"]: row for row in rows if row["eval_lang"] == target}
        writer = SplitWriter.for_pair(self.cfg.prepare.out_dir, target, self.logger)
        with writer.canonical_path(target, TRAIN).open() as lines:
            full = sum(1 for _ in lines)
        curve = (
            [{**main[system_name(ZERO_SHOT)], "lines": 0}] if system_name(ZERO_SHOT) in main else []
        )

        for condition, others in trained.items():
            system = system_name(condition)
            curve.append({**main[system], "lines": full})
            curve.extend(
                {
                    **self._row(system, run, target, self._adapter_scores(run)[target]),
                    "lines": kind.target_lines,
                }
                for run, kind in others
            )

        return curve

    def _adapter_scores(self, run: Path) -> Scores:
        """Get a fine-tuned run's scores on the partition, scoring it only when needed.

        Args:
            run (Path): The run folder.

        Returns:
            Scores: The scores per language.
        """
        return self._ensure(run, lambda: self._score_adapter(run))

    def _ensure(self, run: Path, score: Callable[[], ScoreSummary]) -> dict[str, dict[str, Any]]:
        """Get a run's scores on the partition, scoring it only when needed.

        Args:
            run (Path): The run folder.
            score (Callable[[], ScoreSummary]): The function that scores the run.

        Returns:
            dict[str, dict[str, Any]]: The scores per language.
        """
        path = scores_path(run, self.split, self.max_samples)

        if self.cfg.evaluate.overwrite or not path.exists():
            return score().scores

        self.logger.info(f"Reusing {path}")

        return read_json(path)["scores"]

    def _score_adapter(self, run: Path) -> ScoreSummary:
        """Score one fine-tuned run on the partition.

        Args:
            run (Path): The run folder.

        Returns:
            ScoreSummary: The scores.
        """
        return AdapterScorer(run, self.split, self.max_samples, self.cfg.wandb, self.logger).score()

    def _zero_shot_scorer(self, peft_run: Path) -> PromptScorer:
        """Set up zero-shot scoring of the base model a PEFT run started from.

        Args:
            peft_run (Path): A PEFT run of the evaluated configuration.

        Returns:
            PromptScorer: A scorer that can score the base model on the partition.
        """
        model_cfg = pinned_model_cfg(read_run_info(peft_run))
        prompt_cfg = self.cfg.prompting.copy()
        prompt_cfg.split = self.split
        prompt_cfg.max_samples = self.max_samples
        prompt_cfg.shots = 0

        return PromptScorer(prompt_cfg, model_cfg, self.cfg.wandb, self.logger)

    def _log(self, summary: ResultsSummary) -> None:
        """Log the results tables, and the dev ranking if this run made it, to W&B.

        Args:
            summary (ResultsSummary): The results.
        """
        wandb_run = tracking.start(
            self.cfg.wandb,
            f"{tracking.RESULTS}/{self.split}/{self.run_name}",
            self.run_name,
            tracking.RESULTS,
            {"evaluate": OmegaConf.to_container(self.cfg.evaluate, resolve=True)},
        )

        tracking.log_table(wandb_run, f"{self.split}/results", summary.rows)
        tracking.log_table(wandb_run, f"{self.split}/per_emotion", summary.per_emotion)

        if self.selection is not None:
            tracking.log_table(
                wandb_run,
                "selection/ranking",
                self.selection.ranking(),
            )

        tracking.finish(wandb_run)
