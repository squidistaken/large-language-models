from typing import Any

from matplotlib.figure import Figure

from src.evaluation.metrics import metric_label
from src.plotting.figures.base import BaseFigure
from src.plotting.figures.metrics import HEADLINE_METRICS
from src.plotting.style import UNIT, bars, color_keys, legend, panels


class SeedVariation(BaseFigure):
    """Class for the headline metrics of one configuration over several seeds."""

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Get the seed summary, one row per system and language.

        Returns:
            list[dict[str, Any]]: A list of rows, empty without seed repetitions.
        """
        return self.data.results.get("seeds", [])

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"seeds.{self.data.split}.{self.data.run_name}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        seeds = ", ".join(str(seed) for seed in self.rows[0]["seeds"])

        return (
            f"Headline metrics on {self.data.split}, mean ± sd over seeds {seeds} "
            f"({self.data.run_name})"
        )

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None without seed repetitions.
        """
        if not self.rows:
            return None

        lookup = {(row["system"], row["eval_lang"]): row for row in self.rows}
        colors = self.data.colors({row["system"] for row in self.rows})
        langs = self.data.languages(row["eval_lang"] for row in self.rows)
        fig, axes = panels(1, len(HEADLINE_METRICS), 2.9, sharey=True)

        for ax, metric in zip(axes[0], HEADLINE_METRICS, strict=True):

            def spread(name: str, lang: str, metric: str = metric) -> list[float] | None:
                if (name, lang) not in lookup:
                    return None

                summary = lookup[(name, lang)][metric]

                return [summary["mean"] - summary["std"], summary["mean"] + summary["std"]]

            bars(
                ax,
                langs,
                colors,
                lambda name, lang, metric=metric: (
                    lookup[(name, lang)][metric]["mean"] if (name, lang) in lookup else None
                ),
                interval=spread,
                label="{:.2f}",
            )
            ax.set_title(metric_label(metric))
            ax.set_xlabel("Evaluation language")

        axes[0][0].set_ylabel("Score (mean ± sd)")
        axes[0][0].set_ylim(*UNIT)
        legend(fig, color_keys(colors))

        return fig
