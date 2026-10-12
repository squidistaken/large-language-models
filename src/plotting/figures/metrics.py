import math

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.constants import AGGREGATE_METRICS, HEADLINE_METRICS
from src.evaluation.metrics import metric_label
from src.plotting.figures.base import BaseFigure, cell
from src.plotting.style import UNIT, bars, color_keys, legend, panels

APPENDIX_METRICS = tuple(key for key in AGGREGATE_METRICS if key not in HEADLINE_METRICS)
PANEL_COLUMNS = 3
ZOOMED_METRICS = ("malformed_rate", "empty_rate")
ZOOMED_LIMIT = 0.1


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
        return f"{self.heading} on {self.data.split} {self.context}"

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
        zoomed = [metric for metric in self.metrics if metric in ZOOMED_METRICS]
        fig, axes = panels(panel_rows, PANEL_COLUMNS, 2.5 * panel_rows + 0.4, sharey=not zoomed)

        for ax, metric in zip(axes.flat, self.metrics, strict=False):
            bars(
                ax,
                langs,
                colors,
                cell(lookup, metric),
                interval=cell(lookup, f"{metric}_ci"),
                label="{:.1%}" if metric in zoomed else "{:.2f}",
            )
            ax.set_xlabel("Evaluation language")

            # Rates near zero get their own scale, or their bars would not show on 0 to 1.
            if metric in zoomed:
                top = max([ZOOMED_LIMIT, *(row[metric] * 1.25 for row in rows)])
                ax.set_ylim(0, top)
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
                ax.set_title(f"{metric_label(metric)} (zoomed)")

            else:
                ax.set_ylim(*UNIT)
                ax.set_title(metric_label(metric))

        for ax in axes.flat[len(self.metrics) :]:
            ax.set_visible(False)

        for ax in axes[:, 0]:
            ax.set_ylabel("Score")

        legend(fig, color_keys(colors))

        return fig


class AppendixMetrics(Metrics):
    """Class for the aggregate metrics of every PEFT condition, for the appendix."""

    metrics = APPENDIX_METRICS
    prefix = "metrics_appendix"
    heading = "Other metrics"
