from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import EMOTIONS, TRAIN, ZERO_SHOT
from src.data.prepare_data import pair_name
from src.data.writing import SplitWriter
from src.evaluation.metrics import AGGREGATE_METRICS, bootstrap_intervals, metrics_info
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
    RUN_INFO_FILE,
    pinned_model_cfg,
    predictions_path,
    read_run_info,
    results_stem,
    run_dir,
    scores_path,
    variant,
)

Scores = dict[str, dict[str, Any]]


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


class FinalEvaluator:
    """Class that scores the selected PEFT configuration and the zero-shot baseline on one
    partition, and gathers every score into one results table. Runs of the configuration with
    another seed (<run_name>.seed<seed>) or capped target-language training lines
    (<run_name>.n<lines>) add a seed summary and a learning curve."""

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
                    {
                        "system": system_name(condition),
                        "eval_lang": lang,
                        "emotion": emotion,
                        **metrics["per_emotion"][emotion],
                    }
                    for emotion in EMOTIONS
                )

        repetitions, capped = self._variants()
        summary.seeds = self._seeds(runs, repetitions)
        summary.curve = self._curve(runs, capped, summary.rows)

        stem = results_stem(self.split, self.max_samples, self.run_name)
        summary.files = [str(Path(self.cfg.evaluate.out_dir) / f"{stem}.json")]

        write_json(
            Path(summary.files[0]),
            {
                "run_name": summary.run_name,
                "split": summary.split,
                "max_samples": summary.max_samples,
                "metrics": metrics_info(),
                "results": summary.rows,
                "per_emotion": summary.per_emotion,
                "seeds": summary.seeds,
                "learning_curve": summary.curve,
            },
        )

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

    def _variants(self) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
        """Get the repetitions and capped runs of the selected configuration.

        Returns:
            tuple[dict[str, list[Path]], dict[str, list[Path]]]: A dictionary of repetitions and a
                                                                 dictionary of capped runs, keyed by
                                                                 condition.
        """
        repetitions: dict[str, list[Path]] = {}
        capped: dict[str, list[Path]] = {}

        for condition in conditions(self.cfg.finetune.target_lang):
            for info in sorted(
                Path(self.cfg.finetune.out_dir, condition).glob(f"*/{RUN_INFO_FILE}")
            ):
                config = read_run_info(info.parent)["config"]
                name = variant(config)

                if config["run_name"] != self.run_name or name is None:
                    continue

                if name.startswith("seed") and config.get("max_target_samples") is None:
                    repetitions.setdefault(condition, []).append(info.parent)
                elif name.startswith("n"):
                    capped.setdefault(condition, []).append(info.parent)

        return repetitions, capped

    def _seeds(
        self, runs: dict[str, Path], repetitions: dict[str, list[Path]]
    ) -> list[dict[str, Any]]:
        """Get the mean, standard deviation and values of every metric for every system and
        language, across the main run and its repetitions.

        Args:
            runs (dict[str, Path]): The main run of every condition.
            repetitions (dict[str, list[Path]]): The repetitions per condition.

        Returns:
            list[dict[str, Any]]: A list of results rows, one per system and evaluation language,
                                  with the seeds and the mean, standard deviation and values of
                                  every metric.
        """
        rows = []

        for condition, others in repetitions.items():
            seeded = sorted(
                (read_run_info(run)["config"]["seed"], self._adapter_scores(run))
                for run in [runs[condition], *others]
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

    def _curve(
        self, runs: dict[str, Path], capped: dict[str, list[Path]], rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get the learning curve of the selected configuration, across the main run and its capped
        runs.

        Args:
            runs (dict[str, Path]): The main run of every condition.
            capped (dict[str, list[Path]]): The capped runs per condition.
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
        writer = SplitWriter(
            Path(self.cfg.prepare.out_dir) / pair_name(target), target, self.logger
        )
        full = len(read_jsonl(writer.canonical_path(target, TRAIN)))
        curve = (
            [{**main[system_name(ZERO_SHOT)], "lines": 0}] if system_name(ZERO_SHOT) in main else []
        )

        for condition, others in trained.items():
            system = system_name(condition)
            curve.append({**main[system], "lines": full})
            curve.extend(
                {
                    **self._row(system, run, target, self._adapter_scores(run)[target]),
                    "lines": read_run_info(run)["config"]["max_target_samples"],
                }
                for run in others
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
        score_cfg = OmegaConf.create(
            {"run_dir": str(run), "split": self.split, "max_samples": self.max_samples}
        )

        return AdapterScorer(score_cfg, self.cfg.wandb, self.logger).score()

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
                [
                    {
                        "rank": rank,
                        "run_name": result.run_name,
                        **{
                            f"{lang}_mean": result.language_scores.get(lang)
                            for lang in self.selection.langs
                        },
                        "selection_score": result.selection_score,
                    }
                    for rank, result in enumerate(self.selection.results, start=1)
                ],
            )

        tracking.finish(wandb_run)
