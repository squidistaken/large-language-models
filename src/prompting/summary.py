from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import AGGREGATE_METRICS, PRIMARY_METRICS, RUN_INFO_FILE
from src.evaluation.metrics import metrics_info
from src.utils import tracking
from src.utils.files import write_json
from src.utils.logger import Logger
from src.utils.runs import PromptVariant, read_run_info, read_scores, scores_path, scores_stem

PROMPTING = "prompting"


@dataclass
class PromptingSummary:
    """The summary of prompting runs."""

    split: str
    max_samples: int | None
    models: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    sensitivity: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Get the summary as it is saved.

        Returns:
            dict[str, Any]: A mapping with the settings, the metric definitions, and both tables.
        """
        return {
            "split": self.split,
            "max_samples": self.max_samples,
            "models": self.models,
            "metrics": metrics_info(),
            "results": self.rows,
            "sensitivity": self.sensitivity,
        }


class PromptingSummariser:
    """Class that collects the scores of every prompting run of a language pair."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The full config.
            logger (Logger): The logger.
        """
        self.cfg = cfg
        self.logger = logger
        self.out_dir = Path(cfg.prompting.out_dir)

    def runs(self) -> list[Path]:
        """Get the run folders of every prompting run.

        Returns:
            list[Path]: A list of run folders.
        """
        return sorted(path.parent for path in self.out_dir.glob(f"*/*/{RUN_INFO_FILE}"))

    def models(self) -> list[str]:
        """Get the names of the models with prompting runs.

        Returns:
            list[str]: A list of model config names.
        """
        return sorted({run.name for run in self.runs()})

    def summarise(
        self, split: str, max_samples: int | None, models: list[str] | None = None
    ) -> PromptingSummary:
        """Summarise the prompting runs of the given models, on the given partition, with the given
        number of samples.

        Args:
            split (str): The partition name.
            max_samples (int | None): The number of lines scored per language, or None for all.
            models (list[str] | None): The models to include, or None for every model with runs.

        Returns:
            PromptingSummary: A summary of the runs.

        Raises:
            FileNotFoundError: If no included run has scores on the partition.
            ValueError: If a requested model has no prompting runs.
        """
        runs = self.runs()
        available = sorted({run.name for run in runs})
        models = list(models) if models else available
        unknown = [model for model in models if model not in available]

        if unknown:
            raise ValueError(
                f"No prompting runs in {self.out_dir} for {', '.join(unknown)}. "
                f"Models with runs: {', '.join(available) or 'none'}"
            )

        summary = PromptingSummary(split, max_samples, sorted(models))

        scored = [
            (run, PromptVariant.of(read_run_info(run)["config"]))
            for run in runs
            if run.name in models and scores_path(run, split, max_samples).exists()
        ]

        for run, variant in sorted(
            scored,
            key=lambda item: (
                item[0].name,
                item[1].shots,
                not item[1].is_default,
                item[1].condition,
            ),
        ):
            summary.rows.extend(self._rows(run, variant, read_scores(run, split, max_samples)))

        if not summary.rows:
            raise FileNotFoundError(
                f"No prompting run in {self.out_dir} has {scores_stem(split, max_samples)} scores. "
                "Run `uv run main.py action=prompting` first."
            )

        summary.sensitivity = self._sensitivity(summary.rows)
        out = self.out_dir / f"summary.{scores_stem(split, max_samples)}.json"
        summary.files = [str(out)]

        write_json(out, summary.to_json())
        self._log(summary)
        self.logger.info(f"Summarised {len(summary.rows)} prompting results in {out}")

        return summary

    def _rows(
        self, run: Path, variant: PromptVariant, scores: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get the rows of a prompting run, one per language.

        Args:
            run (Path): The run folder.
            variant (PromptVariant): The run's prompt variant.
            scores (dict[str, dict[str, Any]]): The scores per language.

        Returns:
            list[dict[str, Any]]: A list of rows.
        """
        return [
            {
                "model": run.name,
                "shots": variant.shots,
                "selection": variant.selection_label,
                "template": variant.template,
                "eval_lang": lang,
                **{key: metrics[key] for key in AGGREGATE_METRICS},
                "count": metrics["count"],
            }
            for lang, metrics in scores.items()
        ]

    @staticmethod
    def _sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get the rows of the sensitivity analysis, one per model, shots and language.

        Args:
            rows (list[dict[str, Any]]): The result rows.

        Returns:
            list[dict[str, Any]]: A list of rows, one per model, shots and language.
        """
        groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}

        for row in rows:
            key = (row["model"], row["shots"], row["selection"], row["eval_lang"])
            groups.setdefault(key, []).append(row)

        return [
            {
                "model": model,
                "shots": shots,
                "selection": selection,
                "eval_lang": lang,
                "templates": sorted(row["template"] for row in group),
                **{
                    f"{metric}_{stat}": value
                    for metric in PRIMARY_METRICS
                    for stat, value in _spread([row[metric] for row in group]).items()
                },
            }
            for (model, shots, selection, lang), group in sorted(groups.items())
        ]

    def _log(self, summary: PromptingSummary) -> None:
        """Log both tables to W&B.

        Args:
            summary (PromptingSummary): The summary.
        """
        wandb_run = tracking.start(
            self.cfg.wandb,
            f"{PROMPTING}/{scores_stem(summary.split, summary.max_samples)}",
            PROMPTING,
            "prompting_summary",
            {"prompting": OmegaConf.to_container(self.cfg.prompting, resolve=True)},
        )

        tracking.log_table(wandb_run, f"{summary.split}/prompting", summary.rows)
        tracking.log_table(
            wandb_run,
            f"{summary.split}/prompt_sensitivity",
            [{**row, "templates": ", ".join(row["templates"])} for row in summary.sensitivity],
        )
        tracking.finish(wandb_run)


def _spread(values: list[float]) -> dict[str, float]:
    """Get the mean, standard deviation and range of a list of values.

    Args:
        values (list[float]): The scores.

    Returns:
        dict[str, float]: A mapping with mean, std and range.
    """
    return {"mean": mean(values), "std": pstdev(values), "range": max(values) - min(values)}
