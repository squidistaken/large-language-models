from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from matplotlib.figure import Figure

from src.finetune.conditions import system_name
from src.plotting.figures.base import BaseFigure
from src.plotting.style import HOURS, bars, panels

DEFAULT_VRAM_GIB = 40.0


def hours(run_info: dict[str, Any]) -> float:
    """Get the training time of a run.

    Args:
        run_info (dict[str, Any]): The run's saved run info.

    Returns:
        float: A training time in hours.
    """
    return sum(entry.get("train_runtime", 0.0) for entry in run_info["log_history"]) / 3600


@dataclass(frozen=True)
class CostMeasure:
    """The training cost to compare in the figures."""

    title: str
    axis_label: str
    limits: tuple[float, float]
    value: Callable[[dict[str, Any]], float]
    label: str


def cost_measures(run_infos: Iterable[dict[str, Any]]) -> dict[str, CostMeasure]:
    """Get the training costs to compare in the figures.

    Args:
        run_infos (Iterable[dict[str, Any]]): The run infos of the compared runs.

    Returns:
        dict[str, CostMeasure]: A mapping from cost to measure.
    """
    vram = max(info["model"].get("vram_gb") or DEFAULT_VRAM_GIB for info in run_infos)

    return {
        "memory": CostMeasure(
            "Peak GPU memory",
            "Peak GPU memory (GiB)",
            (0.0, vram),
            lambda info: info["summary"]["peak_gpu_memory_gib"],
            "{:.1f} GiB",
        ),
        "time": CostMeasure("Training time", "Training time (h)", HOURS, hours, "{:.2f} h"),
    }


class Compute(BaseFigure):
    """Class for the training time and peak GPU memory of every PEFT condition."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"compute.{self.data.run_name}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"Training cost per PEFT condition {self.context}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        run_infos = self.data.run_infos
        names = {system_name(condition): info for condition, info in run_infos.items()}
        colors = self.data.colors(list(names))
        measures = cost_measures(run_infos.values())
        fig, axes = panels(1, len(measures), 0.45 * len(names) + 1.2, sharey=True)

        for ax, measure in zip(axes[0], measures.values(), strict=True):
            bars(
                ax,
                list(names),
                colors,
                lambda name, row, value=measure.value: value(names[row]) if name == row else None,
                horizontal=True,
                label=measure.label,
            )
            ax.set_xlim(*measure.limits)
            ax.set_xlabel(measure.axis_label)

        axes[0][0].set_ylabel("PEFT condition")

        return fig
