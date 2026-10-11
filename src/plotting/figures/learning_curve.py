from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.constants import PRIMARY_METRICS, ZERO_SHOT
from src.evaluation.metrics import metric_label
from src.finetune.conditions import system_name
from src.plotting.figures.base import BaseFigure
from src.plotting.style import NEUTRAL, legend, marker_keys, panels


class LearningCurve(BaseFigure):
    """Class for the target-language scores against the number of target-language training
    lines."""

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Get the learning curve, one row per system and size.

        Returns:
            list[dict[str, Any]]: A list of rows, empty without downsampled runs.
        """
        return self.data.results.curve

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"learning_curve.{self.data.stem}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"Learning curve on {self.data.target}, {self.data.split} {self.context}"

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None without downsampled runs.
        """
        baseline = system_name(ZERO_SHOT)
        trained = [row for row in self.rows if row["system"] != baseline]

        if not trained:
            return None

        zero_shot = next((row for row in self.rows if row["system"] == baseline), None)
        names = {row["system"] for row in self.rows}
        colors, markers = self.data.colors(names), self.data.markers(names)
        sizes = sorted({row["lines"] for row in trained})
        fig, axes = panels(1, len(PRIMARY_METRICS), 2.9, sharey=True)

        for ax, metric in zip(axes[0], PRIMARY_METRICS, strict=True):
            for name in colors:
                if name == baseline:
                    continue

                points = sorted(
                    (row["lines"], row[metric], row[f"{metric}_ci"])
                    for row in trained
                    if row["system"] == name
                )
                xs = [x for x, _, _ in points]
                ys = [y for _, y, _ in points]
                errors = [
                    [y - ci[0] for _, y, ci in points],
                    [ci[1] - y for _, y, ci in points],
                ]

                ax.errorbar(
                    xs,
                    ys,
                    yerr=errors,
                    color=colors[name],
                    marker=markers[name],
                    capsize=2,
                    label=name,
                )

            if zero_shot is not None:
                low, high = zero_shot[f"{metric}_ci"]

                ax.axhline(zero_shot[metric], color=colors.get(baseline, NEUTRAL), linestyle="--")
                ax.axhspan(low, high, color=colors.get(baseline, NEUTRAL), alpha=0.12)

            # The capped runs double in size, so a log axis spaces them evenly.
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [str(size) for size in sizes])
            ax.minorticks_off()
            ax.set_xlim(min(sizes) / 1.4, max(sizes) * 1.4)
            ax.set_xlabel(f"Training lines in {self.data.target}")
            ax.set_title(metric_label(metric))

        axes[0][0].set_ylabel(f"Score on {self.data.target}")
        keys = marker_keys({k: v for k, v in colors.items() if k != baseline}, markers)

        if zero_shot is not None:
            keys.insert(
                0,
                Line2D([], [], color=colors.get(baseline, NEUTRAL), linestyle="--", label=baseline),
            )

        legend(fig, keys)

        return fig
