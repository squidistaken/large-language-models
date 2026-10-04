from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.evaluation.metrics import SHORT_NAMES
from src.finetune.conditions import condition_languages, system_name
from src.plotting.figures.base import BaseFigure, FigureData, kept_epoch, series
from src.plotting.figures.metrics import HEADLINE_METRICS
from src.plotting.style import CATEGORICAL, NEUTRAL, legend, panels

LANGUAGE_STYLES = ("-", "--")
# The loss panel stays neutral, so its lines never share a colour with a metric.
DEV_LOSS = "0.55"


class TrainingHistory(BaseFigure):
    """Class for the training history of one PEFT condition of one candidate configuration."""

    def __init__(self, data: FigureData, condition: str, run_name: str) -> None:
        """Initialise the class.

        Args:
            data (FigureData): The inputs of the figures.
            condition (str): The PEFT condition.
            run_name (str): The candidate configuration.
        """
        super().__init__(data)
        self.condition = condition
        self.run_name = run_name

    @classmethod
    def instances(cls, data: FigureData) -> list[BaseFigure]:
        """Get one figure per PEFT condition of every candidate configuration; training happens on
        dev only, so every candidate's history may be shown, not just the selected one.

        Args:
            data (FigureData): The inputs of the figures.

        Returns:
            list[BaseFigure]: A list of figures.
        """
        return [
            cls(data, condition, run_name)
            for run_name in data.candidates
            for condition in data.run_infos
        ]

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"training_history.{self.condition}.{self.run_name}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"Training history of {system_name(self.condition)} ({self.run_name})"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        info = self.data.candidate_infos(self.run_name)[self.condition]
        history = info["log_history"]
        langs = condition_languages(self.condition, self.data.target)
        scored = any(series(history, f"eval_{lang}_micro_f1")[0] for lang in langs)
        kept = kept_epoch(info)
        fig, axes = panels(1, 2 if scored else 1, 2.6)
        loss = axes[0][0]
        epochs, train = series(history, "loss")
        dev_epochs, dev = series(history, "eval_loss")

        loss.plot(epochs, train, color=NEUTRAL, linewidth=1)
        loss.plot(dev_epochs, dev, color=DEV_LOSS, marker="o")
        loss.set_ylim(bottom=0)
        loss.set_xlabel("Epoch")
        loss.set_ylabel("Cross-entropy loss")
        loss.set_title("(a) Loss")

        handles = [
            Line2D([], [], color=NEUTRAL, linewidth=1, label="Training loss"),
            Line2D([], [], color=DEV_LOSS, marker="o", label="Dev loss"),
            Line2D([], [], color=NEUTRAL, linestyle=":", label="Kept checkpoint"),
        ]

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
            handles += [
                *[
                    Line2D([], [], color=color, marker="o", label=SHORT_NAMES[metric])
                    for metric, color in colors.items()
                ],
                *[
                    Line2D([], [], color=NEUTRAL, linestyle=style, label=f"Dev, {lang}")
                    for lang, style in styles.items()
                ],
            ]

        for ax in axes[0]:
            if kept is not None:
                ax.axvline(kept, color=NEUTRAL, linestyle=":", linewidth=1.2)

            ax.set_xlim(left=0)

        legend(fig, handles, ncols=4 if scored else None)

        return fig
