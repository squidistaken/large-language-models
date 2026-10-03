from matplotlib.figure import Figure

from src.constants import SOURCE_LANG
from src.plotting.figures.base import BaseFigure
from src.plotting.style import LANGUAGE_COLORS, UNIT, bars, color_keys, panels


class Selection(BaseFigure):
    """Class for the selection of the best PEFT condition per language."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return "selection"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return "Dev-set selection of the configuration"

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None with fewer than two candidates to compare.
        """
        selection = self.data.selection

        if selection is None or len(selection["results"]) < 2:
            return None

        langs = sorted(selection["langs"], key=lambda lang: lang != SOURCE_LANG)
        scores = {result["run_name"]: result["language_scores"] for result in selection["results"]}
        colors = dict(zip(langs, LANGUAGE_COLORS, strict=False))
        fig, axes = panels(1, 1, 0.4 * len(scores) + 1.2)
        ax = axes[0][0]

        bars(
            ax,
            list(scores),
            colors,
            lambda lang, run: scores[run].get(lang),
            horizontal=True,
            label="{:.3f}",
        )
        ax.set_xlim(*UNIT)
        ax.set_xlabel(f"Mean dev {selection['metric']} over the PEFT conditions")
        ax.set_ylabel("Candidate configuration")
        ax.legend(handles=color_keys(colors), title="Language", loc="lower right")

        return fig
