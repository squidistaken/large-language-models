from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.constants import BILINGUAL, SOURCE_LANG
from src.data.prepare_data import pair_name
from src.data.writing import SplitWriter
from src.evaluation.metrics import example_jaccard
from src.finetune.selection import selected_run
from src.utils.files import read_jsonl, write_jsonl
from src.utils.logger import Logger
from src.utils.runs import predictions_path, read_run_info, run_dir, zero_shot_dir

IMPROVED = "improved"
WORSENED = "worsened"
SAME_CORRECT = "same_correct"
SAME_INCORRECT = "same_incorrect"
CATEGORIES = (IMPROVED, WORSENED, SAME_CORRECT, SAME_INCORRECT)
ALIGNMENT_FIELDS = ("label_source", "alignment_hops", "n_alignments")


def categorise(baseline: float, candidate: float) -> str:
    """Categorise how a line changed between the baseline and the candidate.

    Args:
        baseline (float): The baseline's Jaccard index on the line.
        candidate (float): The candidate's Jaccard index on the line.

    Returns:
        str: A category.
    """
    if candidate > baseline:
        return IMPROVED

    if candidate < baseline:
        return WORSENED

    return SAME_CORRECT if candidate == 1.0 else SAME_INCORRECT


@dataclass
class ComparisonSummary:
    """The outcome of comparing two prediction files line by line."""

    name: str
    baseline: str
    candidate: str
    lang: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


class PredictionComparer:
    """Class that compares two systems' predictions on the same lines."""

    def __init__(self, cfg: DictConfig, logger: Logger, run_name: str | None = None) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The full config.
            logger (Logger): The logger.
            run_name (str | None): The configuration of the default pair, or None for
                evaluate.run_name, else the dev-set selection.
        """
        self.cfg = cfg
        self.logger = logger
        self.lang = str(cfg.analysis.lang)
        self.split = str(cfg.analysis.split)

        if cfg.analysis.baseline and cfg.analysis.candidate:
            self.baseline = Path(cfg.analysis.baseline)
            self.candidate = Path(cfg.analysis.candidate)

        else:
            baseline, candidate = self._defaults(run_name)
            self.baseline = Path(cfg.analysis.baseline or baseline)
            self.candidate = Path(cfg.analysis.candidate or candidate)

    def compare(self) -> ComparisonSummary:
        """Compare the two systems' predictions line by line,

        Returns:
            ComparisonSummary: A summary of the comparison.

        Raises:
            FileNotFoundError: If a prediction file is missing.
            ValueError: If the files share no line.
        """
        for path in (self.baseline, self.candidate):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Score that system first.")

        name = (
            self.cfg.analysis.name
            or f"{self._label(self.baseline)}_vs_{self._label(self.candidate)}"
        )
        summary = ComparisonSummary(
            name=str(name),
            baseline=str(self.baseline),
            candidate=str(self.candidate),
            lang=self.lang,
        )
        baseline = {row["align_id"]: row for row in read_jsonl(self.baseline)}
        records = self._records(self.lang)
        english = self._records(SOURCE_LANG) if self.lang != SOURCE_LANG else {}

        for row in read_jsonl(self.candidate):
            base = baseline.get(row["align_id"])

            if base is None or row["lang"] != self.lang:
                continue

            record = records.get(row["align_id"], {})
            source = english.get(row["align_id"], {})
            baseline_score = example_jaccard(row["gold"], base["predicted"])
            candidate_score = example_jaccard(row["gold"], row["predicted"])

            summary.rows.append(
                {
                    "align_id": row["align_id"],
                    "category": categorise(baseline_score, candidate_score),
                    "subtitle": record.get("text", row["subtitle"]),
                    "gold": row["gold"],
                    "baseline": base["predicted"],
                    "candidate": row["predicted"],
                    "baseline_jaccard": baseline_score,
                    "candidate_jaccard": candidate_score,
                    "baseline_output": base["output"],
                    "candidate_output": row["output"],
                    **{key: record.get(key) for key in ALIGNMENT_FIELDS},
                    "english_text": source.get("text"),
                    "english_labels": source.get("labels"),
                    "english_label_source": source.get("label_source"),
                }
            )

        if not summary.rows:
            raise ValueError(f"{self.baseline} and {self.candidate} share no {self.lang} line.")

        summary.counts = {
            category: sum(row["category"] == category for row in summary.rows)
            for category in CATEGORIES
        }
        path = Path(self.cfg.analysis.out_dir) / f"{summary.name}.jsonl"

        write_jsonl(path, summary.rows)

        summary.files = [str(path)]

        self.logger.info(f"Wrote the comparison to {', '.join(summary.files)}")

        return summary

    def _defaults(self, run_name: str | None) -> tuple[Path, Path]:
        """Get the default baseline and candidate prediction files, based on the selected run.

        Args:
            run_name (str | None): The configuration, or None for the configured or selected one.

        Returns:
            tuple[Path, Path]: A tuple of the baseline and candidate prediction files.

        Raises:
            FileNotFoundError: If no configuration is given and none has been selected yet.
        """
        run_name = run_name or self.cfg.evaluate.run_name or selected_run(self.cfg.finetune.out_dir)

        if run_name is None:
            raise FileNotFoundError(
                "No configuration selected yet. Pass analysis.baseline and analysis.candidate, "
                "or evaluate.run_name=<run_name>."
            )

        bilingual = run_dir(self.cfg.finetune.out_dir, BILINGUAL, str(run_name))
        model_name = read_run_info(bilingual)["model"]["name"]
        zero_shot = zero_shot_dir(self.cfg.prompting.out_dir, model_name)

        return (
            predictions_path(zero_shot, self.split, self.cfg.evaluate.max_samples, self.lang),
            predictions_path(bilingual, self.split, self.cfg.evaluate.max_samples, self.lang),
        )

    def _records(self, lang: str) -> dict[str, dict]:
        """Get the alignment records for one language, keyed by alignment identifier.

        Args:
            lang (str): The language.

        Returns:
            dict[str, dict]: A mapping from alignment identifier to the record, or an empty mapping
                             if the file is missing.
        """
        target = self.cfg.prepare.target_lang
        writer = SplitWriter(
            Path(self.cfg.prepare.out_dir) / pair_name(target), target, self.logger
        )
        path = writer.canonical_path(lang, self.split)

        if not path.exists():
            self.logger.warning(f"Missing {path}; the comparison has no alignment details.")

            return {}

        return {row["align_id"]: row for row in read_jsonl(path)}

    @staticmethod
    def _label(path: Path) -> str:
        """Get a short label for a prediction file, based on its path.

        Args:
            path (Path): The path to the prediction file.

        Returns:
            str: A short label.
        """
        return f"{path.parent.parent.parent.name}-{path.parent.parent.name}"
