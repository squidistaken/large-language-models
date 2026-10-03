from matplotlib.figure import Figure

from src.constants import EMOTIONS
from src.plotting.figures.base import BaseFigure
from src.plotting.style import UNIT, color_keys, dots, language_keys, legend, panels


class PerEmotionF1(BaseFigure):
    """Class for the per-emotion F1 of every PEFT condition."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"per_emotion_f1.{self.data.stem}"

    def draw(self) -> Figure:
        """Draw the figure.

        Returns:
            Figure: A figure.
        """
        rows = self.data.per_emotion
        colors = self.data.colors({row["system"] for row in rows})
        langs = self.data.languages(row["eval_lang"] for row in rows)
        lookup = {(row["system"], row["eval_lang"], row["emotion"]): row for row in rows}
        fig, axes = panels(1, len(langs), 3.2, sharex=True)

        for ax, lang in zip(axes[0], langs, strict=True):
            support = {row["emotion"]: row["support"] for row in rows if row["eval_lang"] == lang}

            dots(
                ax,
                EMOTIONS,
                colors,
                lambda name, emotion, lang=lang: (
                    (lookup[(name, lang, emotion)]["f1"], None)
                    if (name, lang, emotion) in lookup
                    else None
                ),
                lambda _, lang=lang: lang,
            )
            ax.set_yticks(range(len(EMOTIONS)), [f"{e} (n={support[e]:.0f})" for e in EMOTIONS])
            ax.set_title(f"Per-emotion F1, {lang}")
            ax.set_xlabel("F1")

        axes[0][0].set_xlim(*UNIT)
        legend(fig, [*color_keys(colors), *language_keys(langs)])

        return fig
