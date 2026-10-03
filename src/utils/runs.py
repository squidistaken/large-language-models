import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import ZERO_SHOT
from src.utils.files import read_json

RUN_INFO_FILE = "run_info.json"
SCORES_DIR = "scores"
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


def zero_shot_dir(out_dir: str, model_name: str) -> Path:
    """Get the folder holding the zero-shot scores of one model.

    Args:
        out_dir (str): The base folder of the prompting runs.
        model_name (str): The model config name.

    Returns:
        Path: A path.
    """
    return Path(out_dir) / ZERO_SHOT / model_name


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
