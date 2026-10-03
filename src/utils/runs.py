import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.utils.files import read_json

RUN_INFO_FILE = "run_info.json"
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
