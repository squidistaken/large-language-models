import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import RUN_INFO_FILE, SCORES_DIR, ZERO_SHOT
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
    infos = list(Path(out_dir).glob(f"*/*/{RUN_INFO_FILE}"))

    return max(infos, key=lambda path: path.stat().st_mtime).parent if infos else None


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


def zero_shot_dir_of(out_dir: str, peft_run: Path) -> Path:
    """Get the folder holding the zero-shot scores of one PEFT run.

    Args:
        out_dir (str): The base folder of the prompting runs.
        peft_run (Path): The PEFT run folder.

    Returns:
        Path: A path.
    """
    return zero_shot_dir(out_dir, read_run_info(peft_run)["model"]["name"])


def prompting_dir(out_dir: str, condition: str, model_name: str) -> Path:
    """Get the folder holding the scores of one prompting condition of one model.

    Args:
        out_dir (str): The base folder of the prompting runs.
        condition (str): The prompting condition, e.g. zero_shot or 5_shot.
        model_name (str): The model config name.

    Returns:
        Path: A path.
    """
    return Path(out_dir) / condition / model_name


def zero_shot_dir(out_dir: str, model_name: str) -> Path:
    """Get the folder holding the zero-shot scores of one model.

    Args:
        out_dir (str): The base folder of the prompting runs.
        model_name (str): The model config name.

    Returns:
        Path: A path.
    """
    return prompting_dir(out_dir, ZERO_SHOT, model_name)


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
