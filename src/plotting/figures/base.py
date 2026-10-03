from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from omegaconf import DictConfig

from src.constants import SOURCE_LANG, ZERO_SHOT
from src.data.prepare_data import pair_name
from src.data.writing import SplitWriter
from src.finetune.conditions import conditions, system_name
from src.finetune.selection import SELECTION_FILE
from src.plotting.style import NEUTRAL, save, system_colors
from src.utils.files import read_json, read_jsonl
from src.utils.logger import Logger
from src.utils.runs import predictions_path, read_run_info, results_stem, run_dir, zero_shot_dir

CHECKPOINT_METRICS = {"loss": False, "micro_f1": True}
KEPT: dict[str, Any] = {
    "marker": "o",
    "markersize": 11,
    "markerfacecolor": "none",
    "linestyle": "none",
}


class FigureData:
    """Class for the inputs of the report figures."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: Logger,
        run_name: str,
        rows: list[dict[str, Any]] | None = None,
        per_emotion: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The full config.
            logger (Logger): The logger.
            run_name (str): The configuration.
            rows (list[dict[str, Any]] | None): The aggregate results; None reads them from disk.
                                                Defaults to None.
            per_emotion (list[dict[str, Any]] | None): The per-emotion results; None reads them
                                                       from disk. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger
        self.run_name = run_name
        self.target = str(cfg.finetune.target_lang)
        self.split = str(cfg.evaluate.split)
        self.max_samples = cfg.evaluate.max_samples
        self.stem = results_stem(self.split, self.max_samples, run_name)

        if rows is not None and per_emotion is not None:
            self.__dict__["results"] = {"results": rows, "per_emotion": per_emotion}

    @cached_property
    def results(self) -> dict[str, Any]:
        """Get the final results of the configuration.

        Returns:
            dict[str, Any]: A mapping with the aggregate results and the per-emotion results.

        Raises:
            FileNotFoundError: If the results are missing.
        """
        path = Path(self.cfg.evaluate.out_dir) / f"{self.stem}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `uv run main.py action=evaluate_final` first."
            )

        return read_json(path)

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Get the aggregate results, one row per system and language.

        Returns:
            list[dict[str, Any]]: A list of rows.
        """
        return self.results["results"]

    @property
    def per_emotion(self) -> list[dict[str, Any]]:
        """Get the per-emotion results, one row per system, language and emotion.

        Returns:
            list[dict[str, Any]]: A list of rows.
        """
        return self.results["per_emotion"]

    @cached_property
    def runs(self) -> dict[str, Path]:
        """Get the run folder of every PEFT condition.

        Returns:
            dict[str, Path]: A mapping from condition to folder.
        """
        return {
            c: run_dir(self.cfg.finetune.out_dir, c, self.run_name) for c in conditions(self.target)
        }

    @cached_property
    def run_infos(self) -> dict[str, dict[str, Any]]:
        """Get the run info of every PEFT condition.

        Returns:
            dict[str, dict[str, Any]]: A mapping from condition to run info.
        """
        return {condition: read_run_info(path) for condition, path in self.runs.items()}

    @cached_property
    def predictions(self) -> dict[str, dict[str, list[dict]]]:
        """Get the per-line predictions of every system that has them, per language.

        Returns:
            dict[str, dict[str, list[dict]]]: A mapping from system to language to lines.
        """
        model_name = next(iter(self.run_infos.values()))["model"]["name"]
        folders = {ZERO_SHOT: zero_shot_dir(self.cfg.prompting.out_dir, model_name), **self.runs}

        return {
            system_name(condition): {
                lang: read_jsonl(path)
                for lang in (SOURCE_LANG, self.target)
                if (path := predictions_path(folder, self.split, self.max_samples, lang)).exists()
            }
            for condition, folder in folders.items()
        }

    @cached_property
    def splits(self) -> SplitWriter:
        """Get the writer that knows where the prepared splits live.

        Returns:
            SplitWriter: A split writer.
        """
        target = str(self.cfg.prepare.target_lang)

        return SplitWriter(Path(self.cfg.prepare.out_dir) / pair_name(target), target, self.logger)

    def records(self, lang: str, splits: tuple[str, ...]) -> list[dict]:
        """Read the prepared records of one language.

        Args:
            lang (str): The language.
            splits (tuple[str, ...]): The partitions to read.

        Returns:
            list[dict]: A list of records.
        """
        return [
            row for split in splits for row in read_jsonl(self.splits.canonical_path(lang, split))
        ]

    @cached_property
    def selection(self) -> dict[str, Any] | None:
        """Get the saved dev-set selection.

        Returns:
            dict[str, Any] | None: The selection, or None before any selection.
        """
        path = Path(self.cfg.finetune.out_dir) / SELECTION_FILE

        return read_json(path) if path.exists() else None

    def languages(self, present: Iterable[str]) -> list[str]:
        """Get the languages that are present in the results, in display order.

        Args:
            present (Iterable[str]): The language codes present.

        Returns:
            list[str]: A list of language codes.
        """
        found = set(present)

        return [lang for lang in (SOURCE_LANG, self.target) if lang in found]

    def colors(self, names: set[str] | list[str]) -> dict[str, str]:
        """Get the colour of every given system, in display order.

        Args:
            names (set[str] | list[str]): The system names.

        Returns:
            dict[str, str]: A mapping from system name to hex colour.
        """
        return {name: color for name, color in system_colors(self.target).items() if name in names}


class BaseFigure(ABC):
    """Class for the base of the report figures."""

    def __init__(self, data: FigureData) -> None:
        """Initialise the class.

        Args:
            data (FigureData): The inputs of the figures.
        """
        self.data = data

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """

    @property
    @abstractmethod
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """

    @abstractmethod
    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None when there is no data for it.
        """

    def save(self, out_dir: Path, formats: list[str]) -> list[str]:
        """Save the figure to disk, if it can be drawn.

        Args:
            out_dir (Path): The folder to save to.
            formats (list[str]): The file formats.

        Returns:
            list[str]: A list of the saved file paths, or an empty list if the figure could not be
                       drawn.
        """
        fig = self.draw()

        if fig is None:
            return []

        fig.suptitle(self.title)

        return save(fig, out_dir, self.name, formats)


def series(history: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Get the epochs and values of one logged quantity.

    Args:
        history (list[dict[str, Any]]): The Trainer's log history.
        key (str): The logged key.

    Returns:
        tuple[list[float], list[float]]: The epochs and the values.
    """
    points = [(entry["epoch"], entry[key]) for entry in history if key in entry]

    return [epoch for epoch, _ in points], [value for _, value in points]


def kept_epoch(run_info: dict[str, Any]) -> float | None:
    """Get the epoch of the kept checkpoint of a run, if any.

    Args:
        run_info (dict[str, Any]): The run's saved run info.

    Returns:
        float | None: A kept epoch, or None if the run did not keep any checkpoint.
    """
    metric = run_info["config"].get("checkpoint_metric", "loss")
    epochs, values = series(run_info["log_history"], f"eval_{metric}")

    if not values:
        return None

    pick = max if CHECKPOINT_METRICS[metric] else min

    return epochs[values.index(pick(values))]


def kept_key(label: str) -> Line2D:
    """Get the legend entry of the ring marking the kept epoch.

    Args:
        label (str): The legend label.

    Returns:
        Line2D: A legend entry.
    """
    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor=NEUTRAL,
        label=label,
    )
