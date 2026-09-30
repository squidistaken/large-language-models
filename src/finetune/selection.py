from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.finetune.evaluate import SCORES_DIR, evaluation_languages
from src.finetune.trainer import RUN_INFO_FILE
from src.utils.files import read_json, write_json
from src.utils.logger import Logger

SELECTION_FILE = "selection.json"
DEV_SCORES_FILE = "dev.json"

# Settings that must be identical across the three conditions of one candidate, so that each
# candidate really is one shared configuration (and conditions differ only in training data).
SHARED_KEYS = (
    "model_name",
    "model_revision",
    "quantization",
    "lora",
    "learning_rate",
    "epochs",
    "batch_size",
    "grad_accum_steps",
    "max_length",
    "warmup_ratio",
    "checkpoint_metric",
    "seed",
)


@dataclass
class CandidateResult:
    """The dev scores of one candidate configuration."""

    run_name: str
    language_scores: dict[str, float]
    selection_score: float
    settings: dict[str, Any]


@dataclass
class SelectionSummary:
    """The ranked candidates, rendered by the CLI and saved as the selection record."""

    metric: str
    conditions: list[str]
    langs: list[str]
    rule: str
    results: list[CandidateResult] = field(default_factory=list)

    @property
    def best(self) -> CandidateResult:
        """The winning candidate.

        Returns:
            CandidateResult: The candidate with the highest selection score.
        """
        return self.results[0]


class ConfigSelector:
    """Class that picks one final PEFT configuration from scored candidates on the dev set."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune config.
            logger (Logger): The logger.

        Raises:
            ValueError: If no candidates are set.
        """
        if OmegaConf.is_missing(cfg.select, "candidates"):
            raise ValueError(
                "finetune.select.candidates is not set. Pass the run names to compare, e.g. "
                "'finetune.select.candidates=[base,lr_low,rank8]'."
            )

        self.cfg = cfg
        self.logger = logger
        self.out_dir = Path(cfg.out_dir)
        self.conditions = [str(condition) for condition in cfg.select.conditions]
        self.langs = [str(lang) for lang in cfg.select.langs]
        self.metric = str(cfg.select.metric)

    def select(self) -> SelectionSummary:
        """Score every candidate on dev, rank them, and save the selection record.

        Returns:
            SelectionSummary: The ranked candidates.
        """
        rule = (
            f"Highest dev {self.metric}: averaged per language over the conditions "
            f"{', '.join(self.conditions)}, then over the languages {', '.join(self.langs)} "
            "with equal weight. Ties go to the candidate listed first."
        )
        summary = SelectionSummary(
            metric=self.metric, conditions=self.conditions, langs=self.langs, rule=rule
        )

        for run_name in self.cfg.select.candidates:
            summary.results.append(self._evaluate_candidate(str(run_name)))

        summary.results.sort(key=lambda result: result.selection_score, reverse=True)

        write_json(
            self.out_dir / SELECTION_FILE,
            {
                "rule": rule,
                "metric": self.metric,
                "conditions": self.conditions,
                "langs": self.langs,
                "best": summary.best.run_name,
                "ranking": [asdict(result) for result in summary.results],
            },
        )

        self.logger.info(
            f"Selected '{summary.best.run_name}'; saved to {self.out_dir / SELECTION_FILE}"
        )

        return summary

    def _evaluate_candidate(self, run_name: str) -> CandidateResult:
        """Collect and average the dev scores of one candidate across conditions and languages.

        Args:
            run_name (str): The candidate's run name.

        Returns:
            CandidateResult: The candidate's scores.

        Raises:
            FileNotFoundError: If a condition of the candidate has not been trained and scored.
            ValueError: If the candidate has no score in any selection language.
        """
        per_lang: dict[str, list[float]] = {lang: [] for lang in self.langs}
        configs = []

        for condition in self.conditions:
            run_dir = self.out_dir / condition / run_name
            scores_path = run_dir / SCORES_DIR / DEV_SCORES_FILE

            if not scores_path.exists():
                raise FileNotFoundError(
                    f"Missing {scores_path}. Train and score '{run_name}' for condition "
                    f"'{condition}' on dev first."
                )

            configs.append(read_json(run_dir / RUN_INFO_FILE)["config"])
            scores = read_json(scores_path)["scores"]

            for lang in evaluation_languages(condition, self.cfg.target_lang):
                if lang in per_lang:
                    per_lang[lang].append(float(scores[lang][self.metric]))

        self._check_shared(run_name, configs)

        language_scores = {lang: mean(values) for lang, values in per_lang.items() if values}

        if not language_scores:
            raise ValueError(f"Candidate '{run_name}' has no dev score in {', '.join(self.langs)}.")

        return CandidateResult(
            run_name=run_name,
            language_scores=language_scores,
            selection_score=mean(language_scores.values()),
            settings={key: configs[0].get(key) for key in SHARED_KEYS},
        )

    @staticmethod
    def _check_shared(run_name: str, configs: list[dict]) -> None:
        """Check that a candidate used the same settings in every condition.

        Args:
            run_name (str): The candidate's run name.
            configs (list[dict]): The saved config of each condition's run.

        Raises:
            ValueError: If any shared setting differs between conditions.
        """
        first = configs[0]

        for config in configs[1:]:
            different = [key for key in SHARED_KEYS if config.get(key) != first.get(key)]

            if different:
                raise ValueError(
                    f"Candidate '{run_name}' is not one shared configuration: "
                    f"{', '.join(different)} differ between conditions."
                )
