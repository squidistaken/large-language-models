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


class Compute(BaseFigure):
    """Class for the training time and peak GPU memory of every PEFT condition."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"compute.{self.data.run_name}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        run_infos = self.data.run_infos
        names = {system_name(condition): info for condition, info in run_infos.items()}
        colors = self.data.colors(list(names))
        # Memory runs up to the GPU's own capacity, as recorded with the run.
        vram = max(info["model"].get("vram_gb") or DEFAULT_VRAM_GIB for info in run_infos.values())
        measures = {
            "Training time (h)": (HOURS, hours),
            "Peak GPU memory (GiB)": (
                (0.0, vram),
                lambda info: info["summary"]["peak_gpu_memory_gib"],
            ),
        }
        fig, axes = panels(1, len(measures), 0.4 * len(names) + 1.0, sharey=True)

        for ax, (label, (limits, measure)) in zip(axes[0], measures.items(), strict=True):
            bars(
                ax,
                list(names),
                colors,
                lambda name, row, measure=measure: measure(names[row]) if name == row else None,
                horizontal=True,
            )
            ax.set_xlim(*limits)
            ax.set_xlabel(label)

        return fig
