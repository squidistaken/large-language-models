import math

from matplotlib.figure import Figure

from src.evaluation.metrics import AGGREGATE_METRICS, metric_label
from src.plotting.figures.base import BaseFigure
from src.plotting.style import UNIT, bars, color_keys, legend, panels

HEADLINE_METRICS = ("micro_f1", "jaccard_samples", "macro_f1")
APPENDIX_METRICS = tuple(key for key in AGGREGATE_METRICS if key not in HEADLINE_METRICS)
PANEL_COLUMNS = 3


class Metrics(BaseFigure):
    """Class for the aggregate metrics of every PEFT condition."""

    metrics = HEADLINE_METRICS
    prefix = "metrics"
    heading = "Headline metrics"

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"{self.prefix}.{self.data.stem}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"{self.heading} on {self.data.split} ({self.data.run_name})"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        rows = self.data.rows
        colors = self.data.colors({row["system"] for row in rows})
        langs = self.data.languages(row["eval_lang"] for row in rows)
        lookup = {(row["system"], row["eval_lang"]): row for row in rows}
        panel_rows = math.ceil(len(self.metrics) / PANEL_COLUMNS)
        fig, axes = panels(panel_rows, PANEL_COLUMNS, 2.5 * panel_rows + 0.4, sharey=True)

        for ax, metric in zip(axes.flat, self.metrics, strict=False):
            bars(
                ax,
                langs,
                colors,
                lambda name, lang, metric=metric: (
                    lookup[(name, lang)][metric] if (name, lang) in lookup else None
                ),
                interval=lambda name, lang, metric=metric: lookup.get((name, lang), {}).get(
                    f"{metric}_ci"
                ),
                label="{:.2f}",
            )
            ax.set_title(metric_label(metric))
            ax.set_xlabel("Evaluation language")

        for ax in axes.flat[len(self.metrics) :]:
            ax.set_visible(False)

        for ax in axes[:, 0]:
            ax.set_ylabel("Score")

        axes.flat[0].set_ylim(*UNIT)
        legend(fig, color_keys(colors))

        return fig


class AppendixMetrics(Metrics):
    """Class for the aggregate metrics of every PEFT condition, for the appendix."""

    metrics = APPENDIX_METRICS
    prefix = "metrics_appendix"
    heading = "Other metrics"
