import math

from matplotlib.figure import Figure

from src.evaluation.metrics import AGGREGATE_METRICS, metric_label
from src.plotting.figures.base import BaseFigure
from src.plotting.style import UNIT, color_keys, language_keys, legend, marker, offsets, panels

HEADLINE_METRICS = ("micro_f1", "jaccard_samples", "macro_f1")
APPENDIX_METRICS = tuple(key for key in AGGREGATE_METRICS if key not in HEADLINE_METRICS)
PANEL_COLUMNS = 3


class Metrics(BaseFigure):
    """Class for the aggregate metrics of every PEFT condition."""

    metrics = HEADLINE_METRICS
    prefix = "metrics"

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"{self.prefix}.{self.data.stem}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        rows = self.data.rows
        colors = self.data.colors({row["system"] for row in rows})
        langs = self.data.languages(row["eval_lang"] for row in rows)
        lookup = {(row["system"], row["eval_lang"]): row for row in rows}
        names = list(colors)
        panel_rows = math.ceil(len(self.metrics) / PANEL_COLUMNS)
        fig, axes = panels(
            panel_rows,
            PANEL_COLUMNS,
            0.38 * len(names) * panel_rows + 0.9,
            sharex=True,
            sharey=True,
        )

        for ax, metric in zip(axes.flat, self.metrics, strict=False):
            for y, name in enumerate(names):
                found = [(lang, lookup[(name, lang)]) for lang in langs if (name, lang) in lookup]

                if len(found) == 2:
                    values = [row[metric] for _, row in found]
                    ax.plot(values, [y, y], color=colors[name], linewidth=0.8, alpha=0.5)

                # The two languages sit just above and below the row, so their intervals stay apart.
                for (lang, row), offset in zip(found, offsets(len(found)) * 0.5, strict=True):
                    if interval := row.get(f"{metric}_ci"):
                        ax.plot(interval, [y + offset] * 2, color=colors[name], linewidth=1)

                    ax.plot(
                        row[metric], y + offset, "o", markersize=5, **marker(lang, colors[name])
                    )

            ax.set_title(metric_label(metric))
            ax.set_yticks(range(len(names)), names)
            ax.set_ylim(len(names) - 0.5, -0.5)
            ax.grid(axis="y", visible=False)

        for ax in axes.flat[len(self.metrics) :]:
            ax.set_visible(False)

        axes.flat[0].set_xlim(*UNIT)
        legend(fig, [*color_keys(colors), *language_keys(langs)])

        return fig


class AppendixMetrics(Metrics):
    """Class for the aggregate metrics of every PEFT condition, for the appendix."""

    metrics = APPENDIX_METRICS
    prefix = "metrics_appendix"
