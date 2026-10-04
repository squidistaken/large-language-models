from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from src.finetune.selection import resolve_run_name
from src.plotting.figures.alignment import Alignment
from src.plotting.figures.base import BaseFigure, FigureData
from src.plotting.figures.compute import Compute
from src.plotting.figures.configurations import Configurations
from src.plotting.figures.emotion_gain import EmotionGain
from src.plotting.figures.label_counts import LabelCounts
from src.plotting.figures.label_distribution import LabelDistribution
from src.plotting.figures.learning_curve import LearningCurve
from src.plotting.figures.metrics import AppendixMetrics, Metrics
from src.plotting.figures.per_emotion import PerEmotionF1
from src.plotting.figures.seeds import SeedVariation
from src.plotting.figures.selection import Selection
from src.plotting.figures.training_history import TrainingHistory
from src.plotting.style import apply_style
from src.utils.logger import Logger

if TYPE_CHECKING:
    from src.evaluation.results import ResultsSummary

FIGURES: tuple[type[BaseFigure], ...] = (
    TrainingHistory,
    Metrics,
    AppendixMetrics,
    PerEmotionF1,
    EmotionGain,
    LabelDistribution,
    LabelCounts,
    Alignment,
    Compute,
    Selection,
    Configurations,
    SeedVariation,
    LearningCurve,
)


def make_figures(
    cfg: DictConfig,
    logger: Logger,
    run_name: str | None = None,
    summary: ResultsSummary | None = None,
) -> list[str]:
    """Make the report figures of one configuration and write them to disk.

    Args:
        cfg (DictConfig): The full config.
        logger (Logger): The logger.
        run_name (str | None): The configuration; None means evaluate.run_name, else the
                               dev-set selection. Defaults to None.
        summary (ResultsSummary | None): The final results; None reads them from disk.
                                         Defaults to None.

    Returns:
        list[str]: A list of the written figure file paths.

    Raises:
        FileNotFoundError: If no configuration is given and none has been selected yet.
    """
    apply_style()

    data = FigureData(cfg, logger, resolve_run_name(cfg, run_name), summary)
    out_dir = Path(cfg.plot.out_dir)
    formats = [str(fmt) for fmt in cfg.plot.formats]
    files = [
        path
        for kind in FIGURES
        for figure in kind.instances(data)
        for path in figure.save(out_dir, formats)
    ]

    logger.info(f"Wrote the figures to {out_dir}")

    return files
