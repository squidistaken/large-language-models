from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.evaluation.metrics import metric_label
from src.finetune.conditions import condition_languages, system_name
from src.plotting.figures.base import KEPT, BaseFigure, kept_epoch, kept_key, series
from src.plotting.figures.metrics import HEADLINE_METRICS
from src.plotting.style import (
    INK_SECONDARY,
    UNIT,
    condition_color,
    language_keys,
    legend,
    marker,
    panels,
)


class MetricHistory(BaseFigure):
    """Class for the per-epoch dev metrics of every PEFT condition."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"metric_history.{self.data.run_name}"

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None if no run scored its dev subset per epoch.
        """
        target = self.data.target
        runs = self.data.run_infos

        if not any(
            series(info["log_history"], f"eval_{target}_micro_f1")[0] for info in runs.values()
        ):
            return None

        fig, axes = panels(1, len(HEADLINE_METRICS), 2.4, sharey=True)
        max_epoch = 1.0

        for ax, metric in zip(axes[0], HEADLINE_METRICS, strict=True):
            for condition, info in runs.items():
                color = condition_color(condition, target)
                kept = kept_epoch(info)

                for lang in condition_languages(condition, target):
                    epochs, values = series(info["log_history"], f"eval_{lang}_{metric}")
                    max_epoch = max([max_epoch, *epochs])

                    ax.plot(
                        epochs,
                        values,
                        color=color,
                        linewidth=1.2,
                        marker="o",
                        markersize=5,
                        **marker(lang, color),
                    )

                    if kept is not None and kept in epochs:
                        ax.plot(
                            kept, values[epochs.index(kept)], markeredgecolor=INK_SECONDARY, **KEPT
                        )

            ax.set_title(f"Dev {metric_label(metric)}")
            ax.set_xlabel("Epoch")
            ax.grid(axis="x", visible=False)

        for ax in axes[0]:
            ax.set_xlim(0.5, max_epoch + 0.5)
            ax.set_xticks(range(1, int(max_epoch) + 1))

        axes[0][0].set_ylim(*UNIT)
        axes[0][0].set_ylabel("Score")
        legend(
            fig,
            [
                *[
                    Line2D(
                        [],
                        [],
                        color=condition_color(condition, target),
                        label=system_name(condition),
                    )
                    for condition in runs
                ],
                *language_keys(
                    sorted({lang for c in runs for lang in condition_languages(c, target)})
                ),
                kept_key("Kept"),
            ],
        )

        return fig
