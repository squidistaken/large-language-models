import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import (
    COVERAGE,
    DEFAULT_TEMPLATE,
    DEFAULT_VARIANT,
    DEV,
    FULL_PRECISION,
    PINNED,
    RUN_INFO_FILE,
    SCORES_DIR,
    SELECTIONS,
    SHOT_SUFFIX,
    ZERO_SHOT,
)
from src.utils.files import read_json

LIBRARIES = ("torch", "transformers", "peft", "scikit-learn")


def library_versions() -> dict[str, str]:
    """Get the versions of Python and the libraries a run used.

    Returns:
        dict[str, str]: A dictionary containing version strings.
    """
    versions = {"python": platform.python_version()}

    for library in LIBRARIES:
        try:
            versions[library] = version(library)

        except PackageNotFoundError:
            continue

    return versions


def read_run_info(run_dir: Path) -> dict[str, Any]:
    """Read the run info a finished run saved.

    Args:
        run_dir (Path): The run folder.

    Returns:
        dict[str, Any]: A run's information.

    Raises:
        FileNotFoundError: If the run folder has no run info.
    """
    path = run_dir / RUN_INFO_FILE

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Is {run_dir} a finished run?")

    return read_json(path)


def latest_run(out_dir: str) -> Path | None:
    """Find the most recently finished run.

    Args:
        out_dir (str): The folder holding the runs.

    Returns:
        Path | None: A run.
    """
    runs = find_runs(out_dir)

    return max(runs, key=lambda run: (run / RUN_INFO_FILE).stat().st_mtime) if runs else None


def find_runs(out_dir: str | Path) -> list[Path]:
    """Find every finished run folder.

    Args:
        out_dir (str | Path): The folder holding the runs.

    Returns:
        list[Path]: A list of run folders, sorted.
    """
    return sorted(path.parent for path in Path(out_dir).glob(f"*/*/{RUN_INFO_FILE}"))


def run_dir(out_dir: str, condition: str, run_name: str) -> Path:
    """Get the folder of one fine-tuning run.

    Args:
        out_dir (str): The base folder of the fine-tuning runs.
        condition (str): The PEFT condition.
        run_name (str): The run name.

    Returns:
        Path: A path.
    """
    return Path(out_dir) / condition / run_name


@dataclass(frozen=True)
class RunVariant:
    """The variant of a run, if it differs from the main run of its configuration."""

    seed: int | None = None
    target_lines: int | None = None

    @classmethod
    def of(cls, config: Any) -> "RunVariant":
        """Get the variant of a run, if it differs from the main run of its configuration.

        Args:
            config (Any): The run's finetune config.

        Returns:
            RunVariant: A run variant.
        """
        main_seed = config.get("main_seed")
        repeated = main_seed is not None and config["seed"] != main_seed

        return cls(config["seed"] if repeated else None, config.get("max_target_samples"))

    @property
    def is_main(self) -> bool:
        """Get whether the run is the main run of its configuration.

        Returns:
            bool: True if the run is the main run, False if it is a repetition or a capped run.
        """
        return self.seed is None and self.target_lines is None

    @property
    def is_repetition(self) -> bool:
        """Get whether the run repeats its configuration with only another seed.

        Returns:
            bool: True if the run is a repetition, False if it is the main run or a capped run.
        """
        return self.seed is not None and self.target_lines is None

    @property
    def is_capped(self) -> bool:
        """Get whether the run trains its configuration on only fewer target-language lines.

        Returns:
            bool: True if the run is a capped run, False if it is the main run or a repetition.
        """
        return self.target_lines is not None and self.seed is None

    def folder(self, run_name: str) -> str:
        """Get the folder name of a run variant.

        Args:
            run_name (str): The run name of the configuration.

        Returns:
            str: A folder name.
        """
        parts = [f"seed{self.seed}"] if self.seed is not None else []

        if self.target_lines is not None:
            parts.append(f"n{self.target_lines}")

        return ".".join([run_name, *parts])


@dataclass(frozen=True)
class PromptVariant:
    """The variant of a prompting run, if it differs from the main run of its configuration."""

    shots: int = 0
    selection: str = COVERAGE
    template: str = DEFAULT_TEMPLATE

    def __post_init__(self) -> None:
        """Get the demonstration selection and check it.

        Raises:
            ValueError: If the selection is unknown.
        """
        if self.selection not in (*SELECTIONS, PINNED):
            raise ValueError(
                f"Unknown demonstration selection '{self.selection}'. "
                f"Choose one of: {', '.join(SELECTIONS)}"
            )

    @classmethod
    def of(cls, config: Any) -> "PromptVariant":
        """Get the variant of a prompting run, if it differs from the main run of its configuration.

        Args:
            config (Any): The prompting config.

        Returns:
            PromptVariant: A prompt variant.
        """
        pinned = bool(config.get("demonstration_ids"))
        selection = PINNED if pinned else config.get("demonstration_selection") or COVERAGE

        return cls(
            int(config.get("shots") or 0),
            selection,
            str(config.get("template") or DEFAULT_TEMPLATE),
        )

    @property
    def condition(self) -> str:
        """Get the name of the prompting condition, which tells a variant apart from the default run of its configuration.

        Returns:
            str: A condition.
        """
        shots = ZERO_SHOT if not self.shots else f"{self.shots}{SHOT_SUFFIX}"

        return shots if self.is_default else f"{shots}-{self.variant}"

    @property
    def variant(self) -> str:
        """Get the name of the prompting variant.

        Returns:
            str: A variant.
        """
        return "-".join(self._deviations) or DEFAULT_VARIANT

    @property
    def _deviations(self) -> list[str]:
        """Get the design choices that deviate from the default run of a configuration.

        Returns:
            list[str]: A list of deviations.
        """
        deviations = []

        if self.shots and self.selection not in (COVERAGE, PINNED):
            deviations.append(self.selection)

        if self.template != DEFAULT_TEMPLATE:
            deviations.append(self.template)

        return deviations

    @property
    def selection_label(self) -> str:
        """Get the label of the demonstration selection, which tells a variant apart from the
        default run of its configuration.

        Returns:
            str: A label.
        """
        return self.selection if self.shots else "-"

    @property
    def is_default(self) -> bool:
        """Get whether the run is the default run of its configuration, with the default prompt
        template and the default demonstration selection.

        Returns:
            bool: True if neither design choice deviates from its default.
        """
        return not self._deviations

    def check_split(self, split: str) -> None:
        """Check that a sensitivity variant is only scored on dev.

        Args:
            split (str): The partition name.

        Raises:
            ValueError: If a non-default variant would be scored on another partition.
        """
        if not self.is_default and split != DEV:
            raise ValueError(
                f"The prompt variant '{self.condition}' is for the sensitivity analysis on {DEV} "
                f"only; score {split} with the default template and selection."
            )


def model_folder(name: str, quantization: str | None) -> str:
    """Get the folder name of a model, which is the config name plus its quantization if not full
    precision.

    Args:
        name (str): The model config name.
        quantization (str | None): The quantization.

    Returns:
        str: A folder name.
    """
    return name if quantization in FULL_PRECISION else f"{name}.{quantization}"


def prompting_dir(out_dir: str, condition: str, folder: str) -> Path:
    """Get the folder holding the scores of one prompting condition of one model.

    Args:
        out_dir (str): The base folder of the prompting runs.
        condition (str): The prompting condition, e.g. zero_shot or 5_shot.
        folder (str): The model folder, see model_folder.

    Returns:
        Path: A path.
    """
    return Path(out_dir) / condition / folder


def prompting_dir_of(out_dir: str, condition: str, peft_run: Path) -> Path:
    """Get the folder holding the scores of one prompting condition of one model, from a PEFT run.

    Args:
        out_dir (str): The base folder of the prompting runs.
        condition (str): The prompting condition.
        peft_run (Path): The PEFT run folder.

    Returns:
        Path: A path.
    """
    model = read_run_info(peft_run)["model"]

    return prompting_dir(out_dir, condition, model_folder(model["name"], model.get("quantization")))


def scores_stem(split: str, max_samples: int | None) -> str:
    """Get a file name stem for the scores of one partition.

    Args:
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.

    Returns:
        str: A file name stem.
    """
    return split if max_samples is None else f"{split}.sample{max_samples}"


def scores_path(run: Path, split: str, max_samples: int | None) -> Path:
    """Get the scores file of one partition of a run.

    Args:
        run (Path): The run folder.
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.

    Returns:
        Path: A path.
    """
    return run / SCORES_DIR / f"{scores_stem(split, max_samples)}.json"


def read_scores(run: Path, split: str, max_samples: int | None) -> dict[str, dict[str, Any]]:
    """Read the scores of one partition of a run, per language.

    Args:
        run (Path): The run folder.
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.

    Returns:
        dict[str, dict[str, Any]]: A dictionary of scores, one per language.
    """
    return read_json(scores_path(run, split, max_samples))["scores"]


def predictions_path(run: Path, split: str, max_samples: int | None, lang: str) -> Path:
    """Get the per-line predictions of one language and partition of a run.

    Args:
        run (Path): The run folder.
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.
        lang (str): The language.

    Returns:
        Path: A path.
    """
    return run / SCORES_DIR / f"{scores_stem(split, max_samples)}.{lang}.predictions.jsonl"


def results_stem(split: str, max_samples: int | None, run_name: str) -> str:
    """Get the file name stem of the final results tables of one configuration.

    Args:
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.
        run_name (str): The run name of the configuration.

    Returns:
        str: A file name stem.
    """
    return f"{scores_stem(split, max_samples)}.{run_name}"


def results_path(out_dir: str, split: str, max_samples: int | None, run_name: str) -> Path:
    """Get the final results file of one configuration.

    Args:
        out_dir (str): The folder holding the results.
        split (str): The partition name.
        max_samples (int | None): The number of lines scored per language, or None for all.
        run_name (str): The run name of the configuration.

    Returns:
        Path: A path.
    """
    return Path(out_dir) / f"{results_stem(split, max_samples, run_name)}.json"


def pinned_model_cfg(run_info: dict[str, Any]) -> DictConfig:
    """Get the model config of a run, pinned to the exact commit it was trained or scored with.

    Args:
        run_info (dict[str, Any]): The run's saved run info.

    Returns:
        DictConfig: A model config.
    """
    model_cfg = OmegaConf.create(run_info["model"])
    model_cfg.revision = model_cfg.revision_sha

    return model_cfg
