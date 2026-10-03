from matplotlib.figure import Figure

from src.constants import EMOTIONS, ZERO_SHOT
from src.finetune.conditions import system_name
from src.plotting.figures.base import BaseFigure
from src.plotting.style import GAIN, INK_SECONDARY, bars, color_keys, legend, panels


class EmotionGain(BaseFigure):
    """Class for the per-emotion F1 gain of every PEFT condition over zero-shot."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"emotion_gain.{self.data.stem}"

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None without zero-shot results to compare against.
        """
        rows = self.data.per_emotion
        baseline = system_name(ZERO_SHOT)
        f1 = {(row["system"], row["eval_lang"], row["emotion"]): row["f1"] for row in rows}
        langs = [
            lang
            for lang in self.data.languages(row["eval_lang"] for row in rows)
            if (baseline, lang, EMOTIONS[0]) in f1
        ]

        if not langs:
            return None

        colors = self.data.colors({row["system"] for row in rows} - {baseline})
        fig, axes = panels(1, len(langs), 3.4, sharex=True, sharey=True)

        for ax, lang in zip(axes[0], langs, strict=True):

            def gain(name: str, emotion: str, lang: str = lang) -> float | None:
                if (name, lang, emotion) not in f1:
                    return None

                return f1[(name, lang, emotion)] - f1[(baseline, lang, emotion)]

            bars(ax, EMOTIONS, colors, gain, horizontal=True)
            ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
            ax.set_title(f"F1 gain over zero-shot, {lang}")
            ax.set_xlabel("F1 minus zero-shot F1")

        axes[0][0].set_xlim(*GAIN)
        legend(fig, color_keys(colors))

        return fig
