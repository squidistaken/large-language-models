from matplotlib.figure import Figure

from src.constants import EMOTIONS, SPLIT_NAMES
from src.data.prepare_data import pair_name
from src.plotting.figures.base import BaseFigure
from src.plotting.style import INK_SECONDARY, SHARE, dots, language_keys, legend, panels


class LabelDistribution(BaseFigure):
    """Class for the emotion distribution of the prepared lines of every language."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"label_distribution.{pair_name(self.data.target)}"

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

        fig, axes = panels(1, 1, 2.6)
        ax = axes[0][0]

        dots(
            ax,
            EMOTIONS,
            dict.fromkeys(langs, INK_SECONDARY),
            lambda lang, emotion: (shares[lang][emotion], None),
            lambda lang: lang,
        )
        ax.set_xlim(*SHARE)
        ax.set_xlabel("Share of lines with the emotion")
        legend(fig, language_keys(langs))

        return fig
