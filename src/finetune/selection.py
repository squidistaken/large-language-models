from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from omegaconf import DictConfig

from src.constants import DEV, SCORES_DIR, SELECTION_FILE, SOURCE_LANG
from src.finetune.conditions import conditions, evaluation_languages
from src.utils.files import read_json, write_json
from src.utils.logger import Logger
from src.utils.runs import (
    RunVariant,
    read_run_info,
    read_scores,
    run_dir,
    scores_path,
    scores_stem,
)

DEV_SCORES_FILE = f"{scores_stem(DEV, None)}.json"
MODEL_KEYS = (
    "hf_id",
    "revision_sha",
    "dtype",
    "max_new_tokens",
    "decoding",
    "chat_template_kwargs",
)
FIXED_KEYS = (
    "model",
    "data_dir",
    "target_lang",
    "max_train_samples",
    "max_target_samples",
    "max_eval_samples",
    "max_length",
    "seed",
)
SHARED_KEYS = (
    *FIXED_KEYS,
    "lora",
    "quantization",
    "learning_rate",
    "epochs",
    "batch_size",
    "grad_accum_steps",
    "warmup_ratio",
    "evals_per_epoch",
    "checkpoint_metric",
    "early_stopping_patience",
)


def differing_keys(settings: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
    """Get the keys that differ between a list of settings.

    Args:
        settings (list[dict[str, Any]]): The compared settings of each run.
        keys (tuple[str, ...]): The settings that must match.

    Returns:
        list[str]: A list of the keys that differ, empty if all runs match.
    """
    return [key for key in keys if any(other[key] != settings[0][key] for other in settings[1:])]


@dataclass
class CandidateResult:
    """The dev scores of one candidate configuration."""

    run_name: str
    language_scores: dict[str, float]
    selection_score: float
    settings: dict[str, Any]


@dataclass
class SelectionSummary:
    """The ranked candidates."""

    metric: str
    conditions: list[str]
    langs: list[str]
    rule: str
    results: list[CandidateResult] = field(default_factory=list)

    @property
    def best(self) -> CandidateResult:
        """Get the winning candidate.

        Returns:
            CandidateResult: A candidate with the highest selection score.
        """
        return self.results[0]

    def ranking(self) -> list[dict[str, Any]]:
        """Get the ranked candidates as table rows.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, one per candidate, with the rank,
                                  run name, per-language mean and selection score.
        """
        return [
            {
                "rank": rank,
                "run_name": result.run_name,
                **{f"{lang}_mean": result.language_scores.get(lang) for lang in self.langs},
                "selection_score": result.selection_score,
            }
            for rank, result in enumerate(self.results, start=1)
        ]


class ConfigSelector:
    """Class that picks one final PEFT configuration from scored candidates on the dev set."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune config.
            logger (Logger): The logger.

        """
        self.cfg = cfg
        self.logger = logger
        self.out_dir = Path(cfg.out_dir)
        self.conditions = conditions(cfg.target_lang)
        self.langs = [SOURCE_LANG, cfg.target_lang]
        self.metric = str(cfg.select.metric)

    def select(self) -> SelectionSummary:
        """Select the best candidate configuration by averaging its dev scores over the conditions
        and languages.

        Returns:
            SelectionSummary: A summary of the selection process.

        Raises:
            ValueError: If there are no candidates, or they were not trained with the same model
                        and data.
        """
        summary = SelectionSummary(
            metric=self.metric,
            conditions=self.conditions,
            langs=self.langs,
            rule=(
                f"Highest dev {self.metric}: averaged per language over the conditions "
                f"{', '.join(self.conditions)}, then over the languages {', '.join(self.langs)} "
                "with equal weight. Ties go to the candidate listed first."
            ),
        )

        for run_name in self._candidates():
            summary.results.append(self._evaluate_candidate(run_name))

        different = differing_keys([result.settings for result in summary.results], FIXED_KEYS)

        if different:
            raise ValueError(f"The candidates are not comparable: {', '.join(different)} differ.")

        summary.results.sort(key=lambda result: result.selection_score, reverse=True)

        write_json(
            self.out_dir / SELECTION_FILE, {**asdict(summary), "best": summary.best.run_name}
        )

        self.logger.info(
            f"Selected '{summary.best.run_name}'; saved to {self.out_dir / SELECTION_FILE}"
        )

        return summary

    def _candidates(self) -> list[str]:
        """Get the run names to compare.

        Returns:
            list[str]: A list of run names.

        Raises:
            ValueError: If no run is complete in every condition yet.
        """
        if self.cfg.select.candidates:
            return [str(run_name) for run_name in self.cfg.select.candidates]

        complete = set.intersection(
            *(
                {
                    path.parent.parent.name
                    for path in (self.out_dir / condition).glob(f"*/{SCORES_DIR}/{DEV_SCORES_FILE}")
                    if RunVariant.of(read_run_info(path.parent.parent)["config"]).is_main
                }
                for condition in self.conditions
            )
        )

        if not complete:
            raise ValueError(
                "No run has been scored on the full dev set in every condition "
                f"({', '.join(self.conditions)}) yet."
            )

        return sorted(complete)

    def _evaluate_candidate(self, run_name: str) -> CandidateResult:
        """Evaluate one candidate by averaging its dev scores over the conditions and languages.

        Args:
            run_name (str): The candidate's run name.

        Returns:
            CandidateResult: A candidate's score.

        Raises:
            FileNotFoundError: If a condition of the candidate has not been trained and scored.
            ValueError: If the candidate has no score in any selection language.
        """
        per_lang: dict[str, list[float]] = {lang: [] for lang in self.langs}
        configs = []

        for condition in self.conditions:
            run = run_dir(self.cfg.out_dir, condition, run_name)
            path = scores_path(run, DEV, None)

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Train '{run_name}' for condition '{condition}' and score it "
                    "on the full dev set first."
                )

            configs.append(self._settings(read_run_info(run)))
            scores = read_scores(run, DEV, None)

            for lang in evaluation_languages(condition, self.cfg.target_lang):
                if lang in per_lang:
                    per_lang[lang].append(float(scores[lang][self.metric]))

        different = differing_keys(configs, SHARED_KEYS)

        if different:
            raise ValueError(
                f"Candidate '{run_name}' is not one shared configuration: "
                f"{', '.join(different)} differ between conditions."
            )

        language_scores = {lang: mean(values) for lang, values in per_lang.items() if values}

        if not language_scores:
            raise ValueError(f"Candidate '{run_name}' has no dev score in {', '.join(self.langs)}.")

        return CandidateResult(
            run_name=run_name,
            language_scores=language_scores,
            selection_score=mean(language_scores.values()),
            settings=configs[0],
        )

    @staticmethod
    def _settings(run_info: dict[str, Any]) -> dict[str, Any]:
        """Get the settings of one run that selection compares.

        Args:
            run_info (dict[str, Any]): The run's saved run info.

        Returns:
            dict[str, Any]: A dictionary containing the compared settings.
        """
        settings = {
            **run_info["config"],
            "model": {key: run_info["model"].get(key) for key in MODEL_KEYS},
            "quantization": run_info["model"].get("quantization"),
        }

        return {key: settings.get(key) for key in SHARED_KEYS}


def read_selection(out_dir: str) -> dict[str, Any] | None:
    """Read the last dev-set selection.

    Args:
        out_dir (str): The base folder of the fine-tuning runs.

    Returns:
        dict[str, Any] | None: A dictionary with the selection summary, or None if no selection has
                               been made yet.
    """
    path = Path(out_dir) / SELECTION_FILE

    return read_json(path) if path.exists() else None


def selected_run(out_dir: str) -> str | None:
    """Get the run name the last dev-set selection picked.

    Args:
        out_dir (str): The base folder of the fine-tuning runs.

    Returns:
        str | None: A run name, or None if no selection has been made yet.
    """
    selection = read_selection(out_dir)

    return str(selection["best"]) if selection else None


def resolve_run_name(cfg: DictConfig, run_name: str | None = None) -> str:
    """Resolve the run name to score, either from the config or the last selection.

    Args:
        cfg (DictConfig): The full config.
        run_name (str | None): The configuration, or None. Defaults to None.

    Returns:
        str: A run name.

    Raises:
        FileNotFoundError: If no configuration is given and none has been selected yet.
    """
    name = run_name or cfg.evaluate.run_name or selected_run(cfg.finetune.out_dir)

    if name is None:
        raise FileNotFoundError("No configuration selected yet. Pass evaluate.run_name=<run_name>.")

    return str(name)
