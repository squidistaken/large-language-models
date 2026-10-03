from matplotlib.figure import Figure

from src.constants import EMOTIONS, SPLIT_NAMES
from src.data.prepare_data import pair_name
from src.plotting.figures.base import BaseFigure
from src.plotting.style import LANGUAGE_COLORS, SHARE, bars, color_keys, panels


class LabelDistribution(BaseFigure):
    """Class for the emotion distribution of the prepared lines of every language."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"label_distribution.{pair_name(self.data.target)}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return "Emotion distribution of the prepared data"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        langs = list(self.data.splits.languages)
        shares = {}

        for lang in langs:
            lines = [row["labels"] for row in self.data.records(lang, SPLIT_NAMES)]
            shares[lang] = {e: sum(e in line for line in lines) / len(lines) for e in EMOTIONS}

        colors = dict(zip(langs, LANGUAGE_COLORS, strict=False))
        fig, axes = panels(1, 1, 3.0)
        ax = axes[0][0]

        bars(
            ax,
            EMOTIONS,
            colors,
            lambda lang, emotion: shares[lang][emotion],
            horizontal=True,
            label="{:.2f}",
        )
        ax.set_xlim(*SHARE)
        ax.set_xlabel("Share of lines with the emotion")
        ax.set_ylabel("Emotion")
        ax.legend(handles=color_keys(colors), title="Language", loc="lower right")

        return fig
