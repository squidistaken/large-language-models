from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any

from matplotlib.figure import Figure

from src.constants import DEFAULT_VARIANT, PRIMARY_METRICS
from src.evaluation.metrics import metric_label
from src.finetune.conditions import system_name
from src.plotting.figures.base import Drawable, cell, display_languages
from src.plotting.style import (
    MODEL_SHADES,
    NEUTRAL,
    UNIT,
    VARIANT_COLORS,
    VARIANT_MARKERS,
    bars,
    color_keys,
    dots,
    legend,
    marker_keys,
    panels,
)
from src.utils.runs import PromptVariant, scores_stem

if TYPE_CHECKING:
    from src.prompting.summary import PromptingSummary

Row = dict[str, Any]


def group_name(model: str, shots: int) -> str:
    """Get the display name of a model and number of demonstrations.

    Args:
        model (str): The model name.
        shots (int): The number of demonstrations.

    Returns:
        str: A display name.
    """
    return f"{model}, {system_name(PromptVariant(shots=shots).condition)}"


class PromptingFigure(Drawable):
    """Class for the base of every prompting figure."""

    prefix = ""
    heading = ""

    def __init__(self, summary: PromptingSummary) -> None:
        """Initialise the class.

        Args:
            summary (PromptingSummary): The prompting summary.
        """
        self.summary = summary
        self.languages = display_languages(
            (row["eval_lang"] for row in summary.rows), summary.target_lang
        )

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"{self.prefix}.{scores_stem(self.summary.split, self.summary.max_samples)}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"{self.heading} on {self.summary.split}"


class PromptingResults(PromptingFigure):
    """Class for the headline metrics of every model and number of demonstrations, on the default
    prompt variant."""

    prefix = "prompting"
    heading = "Prompting the base models"

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None without runs of the default prompt.
        """
        rows = [row for row in self.summary.rows if row["is_default"]]

        if not rows:
            return None

        present = {(row["model"], row["shots"]) for row in rows}
        models = sorted({model for model, _ in present})
        shots = sorted({count for _, count in present})
        # Only the runs there are: one hue per model, darker with more demonstrations.
        colors = {
            group_name(model, count): shade
            for model, shades in zip(models, cycle(MODEL_SHADES))
            for count, shade in zip(shots, cycle(shades))
            if (model, count) in present
        }
        lookup = {(group_name(row["model"], row["shots"]), row["eval_lang"]): row for row in rows}
        fig, axes = panels(1, len(PRIMARY_METRICS), 2.9, sharey=True)

        for ax, metric in zip(axes[0], PRIMARY_METRICS, strict=True):
            bars(
                ax,
                self.languages,
                colors,
                cell(lookup, metric),
                interval=cell(lookup, f"{metric}_ci"),
                label="{:.2f}",
            )
            ax.set_ylim(*UNIT)
            ax.set_title(metric_label(metric))
            ax.set_xlabel("Evaluation language")

        axes[0][0].set_ylabel("Score")
        # One legend column per model, so each model's runs stack together.
        legend(fig, color_keys(colors), ncols=len(models))

        return fig


class PromptSensitivity(PromptingFigure):
    """Class for the sensitivity of the models to the prompt variant, on the default prompt variant."""

    prefix = "prompt_sensitivity"
    heading = "Prompt sensitivity"

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None when no run has a variant besides the default.
        """
        rows = self.summary.rows
        others = sorted({row["variant"] for row in rows if not row["is_default"]})

        if not others:
            return None

        colors = {DEFAULT_VARIANT: NEUTRAL, **dict(zip(others, cycle(VARIANT_COLORS)))}
        markers = {DEFAULT_VARIANT: "o", **dict(zip(others, cycle(VARIANT_MARKERS)))}
        categories = [
            group_name(model, shots)
            for model, shots in sorted({(row["model"], row["shots"]) for row in rows})
        ]
        lookup = {
            (row["variant"], group_name(row["model"], row["shots"]), row["eval_lang"]): row
            for row in rows
        }
        langs = self.languages
        fig, axes = panels(
            len(langs), len(PRIMARY_METRICS), 0.35 * len(categories) * len(langs) + 1.4, sharey=True
        )

        for lang, row_axes in zip(langs, axes, strict=True):
            for ax, metric in zip(row_axes, PRIMARY_METRICS, strict=True):
                dots(ax, categories, colors, markers, cell(lookup, metric, lang))
                ax.set_title(f"{metric_label(metric)}, {lang}")

        for ax, metric in zip(axes[-1], PRIMARY_METRICS, strict=True):
            ax.set_xlabel(metric_label(metric))

        legend(fig, marker_keys(colors, markers))

        return fig


PROMPTING_FIGURES: tuple[type[PromptingFigure], ...] = (PromptingResults, PromptSensitivity)
