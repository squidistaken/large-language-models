from matplotlib.figure import Figure

from src.constants import EMOTIONS
from src.plotting.figures.base import BaseFigure
from src.plotting.style import UNIT, dots, legend, marker_keys, panels


class PerEmotionF1(BaseFigure):
    """Class for the per-emotion F1 of every PEFT condition."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"per_emotion_f1.{self.data.stem}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"Per-emotion F1 on {self.data.split} {self.context}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        rows = self.data.per_emotion
        names = {row["system"] for row in rows}
        colors, markers = self.data.colors(names), self.data.markers(names)
        langs = self.data.languages(row["eval_lang"] for row in rows)
        lookup = {(row["system"], row["eval_lang"], row["emotion"]): row for row in rows}
        fig, axes = panels(1, len(langs), 3.2, sharex=True)

        for ax, lang in zip(axes[0], langs, strict=True):
            support = {row["emotion"]: row["support"] for row in rows if row["eval_lang"] == lang}

            dots(
                ax,
                EMOTIONS,
                colors,
                markers,
                lambda name, emotion, lang=lang: (
                    lookup[(name, lang, emotion)]["f1"] if (name, lang, emotion) in lookup else None
                ),
            )
            ax.set_yticks(range(len(EMOTIONS)), [f"{e} (n={support[e]:.0f})" for e in EMOTIONS])
            ax.set_title(lang)
            ax.set_xlabel("F1")

        axes[0][0].set_ylabel("Emotion (gold support)")
        axes[0][0].set_xlim(*UNIT)
        legend(fig, marker_keys(colors, markers))

        return fig
