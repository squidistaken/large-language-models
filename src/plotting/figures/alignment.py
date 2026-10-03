from collections import defaultdict
from statistics import mean

import numpy as np
from matplotlib.figure import Figure

from src.evaluation.metrics import BOOTSTRAP_SAMPLES, example_jaccard
from src.plotting.figures.base import BaseFigure
from src.plotting.style import UNIT, color_keys, dots, legend, panels


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
        rows = [f"{hop} hop{'s' if hop != 1 else ''} (n={counts[hop]})" for hop in levels]
        fig, axes = panels(1, 1, 0.45 * len(levels) + 1.0)
        ax = axes[0][0]

        def value(name: str, row: str) -> tuple[float, list[float]] | None:
            lines = scores[name].get(levels[rows.index(row)])

            return (mean(lines), bootstrap_mean(lines, self.data.cfg.seed)) if lines else None

        dots(ax, rows, colors, value, lambda _: target)
        ax.set_xlim(*UNIT)
        ax.set_xlabel(f"Mean per-line Jaccard, {target}")
        legend(fig, color_keys(colors))

        return fig
