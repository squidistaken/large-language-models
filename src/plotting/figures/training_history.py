from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.finetune.conditions import system_name
from src.plotting.figures.base import KEPT, BaseFigure, kept_epoch, kept_key, series
from src.plotting.style import INK_SECONDARY, LOSS, SURFACE, condition_color, legend, panels

SMOOTHING = 0.8
DASHED = (0, (4, 2))


def smooth(values: list[float]) -> list[float]:
    """Smooth a noisy series with a bias-corrected exponential moving average.

    Args:
        values (list[float]): The series.

    Returns:
        list[float]: A smoothed series of the same length.
    """
    smoothed, average = [], 0.0

    for step, value in enumerate(values, start=1):
        average = SMOOTHING * average + (1 - SMOOTHING) * value
        smoothed.append(average / (1 - SMOOTHING**step))

    return smoothed


class TrainingHistory(BaseFigure):
    """Class for the training history of every PEFT condition."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"training_history.{self.data.run_name}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        target = self.data.target
        fig, axes = panels(1, len(self.data.run_infos), 2.3, sharey=True)

        for ax, (condition, run_info) in zip(axes[0], self.data.run_infos.items(), strict=True):
            color = condition_color(condition, target)
            epochs, train = series(run_info["log_history"], "loss")
            dev_epochs, dev = series(run_info["log_history"], "eval_loss")
            kept = kept_epoch(run_info)

            ax.plot(epochs, train, color=color, linewidth=0.7, alpha=0.3)
            ax.plot(epochs, smooth(train), color=color)
            ax.plot(
                dev_epochs,
                dev,
                color=color,
                linestyle=DASHED,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
            )

            if kept is not None:
                ax.plot(kept, dev[dev_epochs.index(kept)], markeredgecolor=INK_SECONDARY, **KEPT)

            ax.set_title(system_name(condition))
            ax.set_xlabel("Epoch")
            ax.set_xlim(left=0)
            ax.grid(axis="x", visible=False)

        axes[0][0].set_ylim(*LOSS)
        axes[0][0].set_ylabel("Cross-entropy loss")
        legend(
            fig,
            [
                Line2D([], [], color=INK_SECONDARY, label="Training loss (smoothed)"),
                Line2D(
                    [],
                    [],
                    color=INK_SECONDARY,
                    linestyle=DASHED,
                    marker="o",
                    markersize=5,
                    markeredgecolor=SURFACE,
                    label="Dev loss (per epoch)",
                ),
                kept_key("Kept checkpoint"),
            ],
        )

        return fig
