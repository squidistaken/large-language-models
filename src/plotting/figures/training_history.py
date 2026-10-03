from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.evaluation.metrics import SHORT_NAMES
from src.finetune.conditions import condition_languages, system_name
from src.plotting.figures.base import BaseFigure, FigureData, kept_epoch, series
from src.plotting.figures.metrics import HEADLINE_METRICS
from src.plotting.style import CATEGORICAL, NEUTRAL, panels

SMOOTHING = 0.8
LANGUAGE_STYLES = ("-", "--")


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
    """Class for the training history of one PEFT condition."""

    def __init__(self, data: FigureData, condition: str) -> None:
        """Initialise the class.

        Args:
            data (FigureData): The inputs of the figures.
            condition (str): The PEFT condition.
        """
        super().__init__(data)
        self.condition = condition

    @classmethod
    def instances(cls, data: FigureData) -> list[BaseFigure]:
        """Get one figure per PEFT condition.

        Args:
            data (FigureData): The inputs of the figures.

        Returns:
            list[BaseFigure]: A list of figures.
        """
        return [cls(data, condition) for condition in data.run_infos]

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"training_history.{self.condition}.{self.data.run_name}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"Training history of {system_name(self.condition)}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        info = self.data.run_infos[self.condition]
        history = info["log_history"]
        langs = condition_languages(self.condition, self.data.target)
        scored = any(series(history, f"eval_{lang}_micro_f1")[0] for lang in langs)
        kept = kept_epoch(info)
        fig, axes = panels(1, 2 if scored else 1, 2.6)
        loss = axes[0][0]
        epochs, train = series(history, "loss")
        dev_epochs, dev = series(history, "eval_loss")

        loss.plot(epochs, train, color=CATEGORICAL[0], linewidth=0.8, alpha=0.3)
        loss.plot(epochs, smooth(train), color=CATEGORICAL[0], label="Training loss (smoothed)")
        loss.plot(dev_epochs, dev, color=CATEGORICAL[1], marker="o", label="Dev loss")
        loss.set_ylim(bottom=0)
        loss.set_xlabel("Epoch")
        loss.set_ylabel("Cross-entropy loss")
        loss.set_title("(a) Loss")

        if scored:
            metrics = axes[0][1]
            colors = dict(zip(HEADLINE_METRICS, CATEGORICAL, strict=False))
            styles: dict[str, Any] = dict(zip(langs, LANGUAGE_STYLES, strict=False))

            for metric, color in colors.items():
                for lang, style in styles.items():
                    points, values = series(history, f"eval_{lang}_{metric}")
                    metrics.plot(points, values, color=color, linestyle=style, marker="o")

            metrics.set_xlabel("Epoch")
            metrics.set_ylabel("Dev score")
            metrics.set_title("(b) Dev metrics")
            metrics.legend(
                handles=[
                    *[
                        Line2D([], [], color=color, marker="o", label=SHORT_NAMES[metric])
                        for metric, color in colors.items()
                    ],
                    *[
                        Line2D([], [], color=NEUTRAL, linestyle=style, label=lang)
                        for lang, style in styles.items()
                    ],
                ],
                loc="best",
                ncols=2,
            )

        for ax in axes[0]:
            if kept is not None:
                ax.axvline(kept, color=NEUTRAL, linestyle=":", linewidth=1.2)

            ax.set_xlim(left=0)

        kept_line = Line2D([], [], color=NEUTRAL, linestyle=":", label="Kept checkpoint")
        loss.legend(handles=[*loss.get_legend_handles_labels()[0], kept_line], loc="best")

        return fig
