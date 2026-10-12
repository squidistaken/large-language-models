from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.constants import CHECKPOINT_METRICS, SOURCE_LANG
from src.data.writing import SplitWriter
from src.evaluation.metrics import eval_key
from src.finetune.conditions import conditions, prompt_baselines, system_name
from src.finetune.selection import read_selection
from src.plotting.style import save, system_colors, system_markers
from src.utils.files import read_jsonl
from src.utils.logger import Logger
from src.utils.runs import (
    model_folder,
    predictions_path,
    prompting_dir,
    read_run_info,
    read_scores,
    results_path,
    results_stem,
    run_dir,
)

if TYPE_CHECKING:
    from src.evaluation.results import ResultsSummary


class FigureData:
    """Class for the inputs of the report figures."""

    def __init__(
        self,
        cfg: DictConfig,
        logger: Logger,
        run_name: str,
        summary: ResultsSummary | None = None,
    ) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The full config.
            logger (Logger): The logger.
            run_name (str): The configuration.
            summary (ResultsSummary | None): The final results; None reads them from disk.
                                             Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger
        self.run_name = run_name
        self.target = str(cfg.finetune.target_lang)
        self.split = str(cfg.evaluate.split)
        self.max_samples = cfg.evaluate.max_samples
        self.stem = results_stem(self.split, self.max_samples, run_name)
        self.summary = summary
        self._infos: dict[str, dict[str, dict[str, Any]]] = {}

    @cached_property
    def results(self) -> ResultsSummary:
        """Get the final results of the configuration.

        Returns:
            ResultsSummary: The results.

        Raises:
            FileNotFoundError: If the results are missing.
        """
        if self.summary is not None:
            return self.summary

        from src.evaluation.results import ResultsSummary

        path = results_path(self.cfg.evaluate.out_dir, self.split, self.max_samples, self.run_name)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `uv run main.py action=evaluate_final` first."
            )

        return ResultsSummary.load(path)

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Get the aggregate results, one row per system and language.

        Returns:
            list[dict[str, Any]]: A list of rows.
        """
        return self.results.rows

    @property
    def per_emotion(self) -> list[dict[str, Any]]:
        """Get the per-emotion results, one row per system, language and emotion.

        Returns:
            list[dict[str, Any]]: A list of rows.
        """
        return self.results.per_emotion

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
        return self.candidate_infos(self.run_name)

    @cached_property
    def model(self) -> str:
        """Get the base model of the evaluated configuration, as named in the prompting folders.

        Returns:
            str: A model name, e.g. eurollm_9b or eurollm_9b.4bit.
        """
        return self.model_of(self.run_name)

    def model_of(self, run_name: str) -> str:
        """Get the base model of one candidate configuration, as named in the prompting folders.

        Args:
            run_name (str): The candidate's run name.

        Returns:
            str: A model name, e.g. eurollm_9b or eurollm_9b.4bit.
        """
        model = next(iter(self.candidate_infos(run_name).values()))["model"]

        return model_folder(model["name"], model.get("quantization"))

    @cached_property
    def candidates(self) -> list[str]:
        """Get the candidate configurations the dev-set selection ranked.

        Returns:
            list[str]: A list of run names.
        """
        if self.selection is None:
            return [self.run_name]

        return [result["run_name"] for result in self.selection["results"]]

    def candidate_infos(self, run_name: str) -> dict[str, dict[str, Any]]:
        """Get the run info of every PEFT condition of one candidate configuration.

        Args:
            run_name (str): The candidate's run name.

        Returns:
            dict[str, dict[str, Any]]: A mapping from condition to run info.
        """
        if run_name not in self._infos:
            self._infos[run_name] = {
                condition: read_run_info(run_dir(self.cfg.finetune.out_dir, condition, run_name))
                for condition in conditions(self.target)
            }

        return self._infos[run_name]

    def scores(self, run_name: str, condition: str, split: str) -> dict[str, dict[str, Any]]:
        """Get the scores of one candidate configuration, condition and split.

        Args:
            run_name (str): The candidate's run name.
            condition (str): The PEFT condition.
            split (str): The partition name.

        Returns:
            dict[str, dict[str, Any]]: A mapping from language to scores.
        """
        run = run_dir(self.cfg.finetune.out_dir, condition, run_name)

        return read_scores(run, split, None)

    @cached_property
    def predictions(self) -> dict[str, dict[str, list[dict]]]:
        """Get the per-line predictions of every system that has them, per language.

        Returns:
            dict[str, dict[str, list[dict]]]: A mapping from system to language to lines.
        """
        folders = {
            **{
                condition: prompting_dir(self.cfg.prompting.out_dir, condition, self.model)
                for condition in prompt_baselines(self.cfg.evaluate.prompt_shots)
            },
            **self.runs,
        }

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
        return SplitWriter.for_pair(
            self.cfg.prepare.out_dir, str(self.cfg.prepare.target_lang), self.logger
        )

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
        return read_selection(self.cfg.finetune.out_dir)

    def languages(self, present: Iterable[str]) -> list[str]:
        """Get the languages that are present in the results, in display order.

        Args:
            present (Iterable[str]): The language codes present.

        Returns:
            list[str]: A list of language codes.
        """
        return display_languages(present, self.target)

    def colors(self, names: set[str] | list[str]) -> dict[str, str]:
        """Get the colour of every given system, in display order.

        Args:
            names (set[str] | list[str]): The system names.

        Returns:
            dict[str, str]: A mapping from system name to hex colour.
        """
        return {
            name: color
            for name, color in system_colors(self.target, self.cfg.evaluate.prompt_shots).items()
            if name in names
        }

    def markers(self, names: set[str] | list[str]) -> dict[str, str]:
        """Get the marker shape of every given system, in display order.

        Args:
            names (set[str] | list[str]): The system names.

        Returns:
            dict[str, str]: A mapping from system name to marker shape.
        """
        return {
            k: v
            for k, v in system_markers(self.target, self.cfg.evaluate.prompt_shots).items()
            if k in names
        }


def display_languages(present: Iterable[str], target: str) -> list[str]:
    """Get the languages that are present in the results, in display order: English, then the
    target language.

    Args:
        present (Iterable[str]): The language codes present.
        target (str): The target language.

    Returns:
        list[str]: A list of language codes.
    """
    found = set(present)

    return [lang for lang in (SOURCE_LANG, target) if lang in found]


def cell(lookup: dict[tuple, dict[str, Any]], key: str, *fixed: str) -> Callable[..., Any]:
    """Get a function that looks up one value of the result rows, or None when there is no row, as
    the value of a series in a category of a panel.

    Args:
        lookup (dict[tuple, dict[str, Any]]): The rows, keyed by series and category, then the
                                              fixed parts of the key.
        key (str): The metric key.
        *fixed (str): The parts of the lookup key after the series and category.

    Returns:
        Callable[..., Any]: A function from a series and category to the value.
    """
    return lambda *parts: lookup.get((*parts, *fixed), {}).get(key)


class Drawable(ABC):
    """Class for the base of every figure that can be drawn and saved to disk."""

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


class BaseFigure(Drawable):
    """Class for the base of every figure that can be drawn and saved to disk, with a title and
    context.
    """

    @classmethod
    def instances(cls, data: FigureData) -> list[BaseFigure]:
        """Get the instances of one figure class for one configuration.

        Args:
            data (FigureData): The inputs of the figures.

        Returns:
            list[BaseFigure]: A list of figures.
        """
        return [cls(data)]

    def __init__(self, data: FigureData) -> None:
        """Initialise the class.

        Args:
            data (FigureData): The inputs of the figures.
        """
        self.data = data

    @property
    def context(self) -> str:
        """Get what the figure's results belong to, for the end of its title.

        Returns:
            str: A context.
        """
        return f"for {self.data.model} ({self.data.run_name})"


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
    epochs, values = series(run_info["log_history"], eval_key(metric))

    if not values:
        return None

    pick = max if CHECKPOINT_METRICS[metric] else min

    return epochs[values.index(pick(values))]
