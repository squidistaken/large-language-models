from collections import Counter

from matplotlib.figure import Figure

from src.plotting.figures.base import BaseFigure
from src.plotting.style import NEUTRAL, UNIT, bars, color_keys, legend, panels

GOLD_SERIES = "gold"
COUNTS = ("0", "1", "2", "3", "4+")


def shares(lines: list[list[str]]) -> dict[str, float]:
    """Get the share of lines with each number of labels.

    Args:
        lines (list[list[str]]): The labels of every line.

    Returns:
        dict[str, float]: A mapping from label-set size to the share of lines.
    """
    counts = Counter(COUNTS[min(len(line), len(COUNTS) - 1)] for line in lines)

    return {size: counts[size] / len(lines) for size in COUNTS}


class LabelCounts(BaseFigure):
    """Class for the number of labels per line of the gold labels and every system."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"label_counts.{self.data.stem}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"Labels per line on {self.data.split} {self.context}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        predictions = self.data.predictions
        gold = {
            lang: [line["gold"] for line in lines]
            for by_lang in predictions.values()
            for lang, lines in by_lang.items()
        }
        colors = {GOLD_SERIES: NEUTRAL, **self.data.colors(list(predictions))}
        langs = self.data.languages(gold)
        fig, axes = panels(1, len(langs), 2.6, sharey=True)

        for ax, lang in zip(axes[0], langs, strict=True):
            found = {GOLD_SERIES: shares(gold[lang])} | {
                name: shares([line["predicted"] for line in by_lang[lang]])
                for name, by_lang in predictions.items()
                if lang in by_lang
            }

            bars(
                ax,
                COUNTS,
                colors,
                lambda name, size, found=found: found[name][size] if name in found else None,
            )
            ax.set_title(lang)
            ax.set_xlabel("Labels on the line")

        axes[0][0].set_ylim(*UNIT)
        axes[0][0].set_ylabel("Share of lines")
        legend(fig, color_keys(colors))

        return fig
