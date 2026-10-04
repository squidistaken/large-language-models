from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.finetune.selection import selected_run
from src.plotting.figures.alignment import Alignment
from src.plotting.figures.base import BaseFigure, FigureData
from src.plotting.figures.compute import Compute
from src.plotting.figures.configurations import Configurations
from src.plotting.figures.emotion_gain import EmotionGain
from src.plotting.figures.label_counts import LabelCounts
from src.plotting.figures.label_distribution import LabelDistribution
from src.plotting.figures.metrics import AppendixMetrics, Metrics
from src.plotting.figures.per_emotion import PerEmotionF1
from src.plotting.figures.selection import Selection
from src.plotting.figures.training_history import TrainingHistory
from src.plotting.style import apply_style
from src.utils.logger import Logger

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
)


def make_figures(
    cfg: DictConfig,
    logger: Logger,
    run_name: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    per_emotion: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Make the report figures of one configuration and write them to disk.

    Args:
        cfg (DictConfig): The full config.
        logger (Logger): The logger.
        run_name (str | None): The configuration; None means evaluate.run_name, else the
                               dev-set selection. Defaults to None.
        rows (list[dict[str, Any]] | None): The aggregate results; None reads them from disk.
                                            Defaults to None.
        per_emotion (list[dict[str, Any]] | None): The per-emotion results; None reads them from
                                                   disk. Defaults to None.

    Returns:
        list[str]: A list of the written figure file paths.

    Raises:
        FileNotFoundError: If no configuration is given and none has been selected yet.
    """
    run_name = run_name or cfg.evaluate.run_name or selected_run(cfg.finetune.out_dir)

    if run_name is None:
        raise FileNotFoundError("No configuration selected yet. Pass evaluate.run_name=<name>.")

    apply_style()

    data = FigureData(cfg, logger, str(run_name), rows, per_emotion)
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
