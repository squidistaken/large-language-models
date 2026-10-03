from collections import defaultdict
from statistics import mean

import numpy as np
from matplotlib.figure import Figure

from src.evaluation.metrics import BOOTSTRAP_SAMPLES, example_jaccard
from src.plotting.figures.base import BaseFigure
from src.plotting.style import UNIT, bars, color_keys, panels


def bootstrap_mean(values: list[float], seed: int) -> list[float]:
    """Get the 95% confidence interval of the mean of a list of values by bootstrapping.

    Args:
        values (list[float]): The values.
        seed (int): The seed of the resampling.

    Returns:
        list[float]: A list of the [lower, upper] bounds.
    """
    data = np.asarray(values)
    resamples = np.random.default_rng(seed).integers(0, len(data), (BOOTSTRAP_SAMPLES, len(data)))
    means = data[resamples].mean(axis=1)

    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


class Alignment(BaseFigure):
    """Class for the per-line Jaccard of every system by alignment hops."""

    @property
    def name(self) -> str:
        """Get the file name stem.

        Returns:
            str: A file name stem.
        """
        return f"alignment.{self.data.stem}"

    @property
    def title(self) -> str:
        """Get the title shown above the figure.

        Returns:
            str: A title.
        """
        return f"Per-line Jaccard by alignment hops on {self.data.split}"

    def draw(self) -> Figure | None:
        """Draw the figure.

        Returns:
            Figure | None: A figure, or None without target-language predictions.
        """
        target = self.data.target
        hops = {
            row["align_id"]: row["alignment_hops"]
            for row in self.data.records(target, (self.data.split,))
        }
        scores: dict[str, dict[int, list[float]]] = {}

        for name, by_lang in self.data.predictions.items():
            if target in by_lang:
                grouped: dict[int, list[float]] = defaultdict(list)

                for line in by_lang[target]:
                    score = example_jaccard(line["gold"], line["predicted"])
                    grouped[hops[line["align_id"]]].append(score)

                scores[name] = dict(grouped)

        if not scores:
            return None

        colors = self.data.colors(list(scores))
        levels = sorted({hop for by_hop in scores.values() for hop in by_hop})
        counts = {
            hop: max(len(by_hop.get(hop, [])) for by_hop in scores.values()) for hop in levels
        }
        groups = [f"{hop} (n={counts[hop]})" for hop in levels]
        level = dict(zip(groups, levels, strict=True))
        fig, axes = panels(1, 1, 2.8)
        ax = axes[0][0]

        bars(
            ax,
            groups,
            colors,
            lambda name, group: (
                mean(scores[name][level[group]]) if scores[name].get(level[group]) else None
            ),
            interval=lambda name, group: (
                bootstrap_mean(scores[name][level[group]], self.data.cfg.seed)
                if scores[name].get(level[group])
                else None
            ),
            label="{:.2f}",
        )
        ax.set_ylim(*UNIT)
        ax.set_xlabel("Alignment hops behind the labels (lines)")
        ax.set_ylabel(f"Mean per-line Jaccard, {target}")
        ax.legend(handles=color_keys(colors), loc="upper right")

        return fig
