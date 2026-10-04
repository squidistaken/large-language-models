import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.constants import DEV
from src.finetune.conditions import evaluation_languages, system_name
from src.plotting.figures.base import BaseFigure
from src.plotting.figures.compute import cost_measures
from src.plotting.style import (
    CONFIGURATION_COLORS,
    TEXT_WIDTH_IN,
    UNIT,
    bars,
    color_keys,
    legend,
)

LABELLED_CANDIDATES = 3


class Configurations(BaseFigure):
    """Class for the comparison of the candidate configurations on dev and in training cost."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return "configurations"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return "Candidate PEFT configurations on dev"

    def draw(self) -> Figure | None:
        """Draw the figure: (a) the dev micro-F1 of every condition and language, (b) the peak GPU
        memory and (c) the training time of every condition, a bar per candidate configuration.

        Returns:
            Figure | None: A figure, or None with fewer than two candidates to compare.
        """
        candidates = self.data.candidates

        if len(candidates) < 2:
            return None

        target = self.data.target
        infos = {run: self.data.candidate_infos(run) for run in candidates}
        colors = dict(zip(candidates, CONFIGURATION_COLORS[: len(candidates)], strict=True))
        conditions = list(infos[candidates[0]])
        dev = {
            (run, condition): self.data.scores(run, condition, DEV)
            for run in candidates
            for condition in conditions
        }
        rows = {
            f"{system_name(condition)}, {lang}": (condition, lang)
            for condition in conditions
            for lang in evaluation_languages(condition, target)
        }
        names = {system_name(condition): condition for condition in conditions}
        measures = cost_measures(
            info for by_condition in infos.values() for info in by_condition.values()
        )
        fig = plt.figure(figsize=(TEXT_WIDTH_IN, 3.2 + 0.4 * len(candidates)), layout="constrained")
        axes = fig.subplot_mosaic([["scores", "scores"], ["memory", "time"]])

        bars(
            axes["scores"],
            list(rows),
            colors,
            lambda run, row: dev[(run, rows[row][0])][rows[row][1]]["micro_f1"],
            horizontal=True,
            label="{:.3f}" if len(candidates) <= LABELLED_CANDIDATES else None,
        )
        axes["scores"].set_xlim(*UNIT)
        axes["scores"].set_xlabel("Dev micro-F1")
        axes["scores"].set_title("(a) Dev micro-F1")

        for (key, measure), letter in zip(measures.items(), "bc", strict=True):
            bars(
                axes[key],
                list(names),
                colors,
                lambda run, row, value=measure.value: value(infos[run][names[row]]),
                horizontal=True,
                label=measure.label,
            )
            axes[key].set_xlim(*measure.limits)
            axes[key].set_xlabel(measure.axis_label)
            axes[key].set_title(f"({letter}) {measure.title}")

        axes["time"].tick_params(labelleft=False)
        legend(fig, color_keys(colors))

        return fig
