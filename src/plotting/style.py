from collections.abc import Callable, Sequence
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.finetune.conditions import conditions, prompt_baselines, system_name

TEXT_WIDTH_IN = 17 / 2.54
CATEGORICAL = ("tab:blue", "tab:orange", "tab:green", "tab:red")
# The PEFT conditions keep fixed colours and markers; the prompting baselines take theirs in turn.
PROMPT_COLORS = ("tab:blue", "tab:cyan", "tab:olive", "tab:gray")
PROMPT_MARKERS = ("o", "v", "P", "X")
PEFT_COLORS = ("tab:orange", "tab:green", "tab:red")
PEFT_MARKERS = ("s", "^", "D")
MODEL_SHADES = (
    ("#9ecae1", "#2171b5"),
    ("#fdae6b", "#d94801"),
    ("#a1d99b", "#238b45"),
    ("#bcbddc", "#6a51a3"),
)
VARIANT_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown")
VARIANT_MARKERS = ("s", "^", "v", "P", "X")
NEUTRAL = "black"
LANGUAGE_COLORS = ("tab:purple", "tab:brown")
CONFIGURATION_COLORS = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
)

UNIT = (0.0, 1.0)  # every metric and share bounded by 0 and 1
GAIN = (-0.3, 0.3)  # difference in F1
SHARE = (0.0, 0.5)  # share of lines carrying one emotion
HOURS = (0.0, 2.0)  # training time per PEFT condition
GROUP_SHARE = 0.8
MAX_UPRIGHT_LABELS = 3
MAX_LEGEND_COLUMNS = 6


def system_styles(
    target_lang: str, prompt_shots: Sequence[int], prompt: Sequence[str], peft: Sequence[str]
) -> dict[str, str]:
    """Get the fixed style of every system in the results, for either colour or marker.

    Args:
        target_lang (str): The assigned target language.
        prompt_shots (Sequence[int]): The number of demonstrations of each prompting baseline.
        prompt (Sequence[str]): The styles of the prompting baselines, used in turn.
        peft (Sequence[str]): The styles of the PEFT conditions.

    Returns:
        dict[str, str]: A mapping from system name to style.
    """
    baselines = [system_name(condition) for condition in prompt_baselines(prompt_shots)]
    trained = [system_name(condition) for condition in conditions(target_lang)]

    return {**dict(zip(baselines, cycle(prompt))), **dict(zip(trained, peft, strict=True))}


def system_colors(target_lang: str, prompt_shots: Sequence[int]) -> dict[str, str]:
    """Get the fixed colour of every system in the results.

    Args:
        target_lang (str): The assigned target language.
        prompt_shots (Sequence[int]): The number of demonstrations of each prompting baseline.

    Returns:
        dict[str, str]: A mapping from system name to colour.
    """
    return system_styles(target_lang, prompt_shots, PROMPT_COLORS, PEFT_COLORS)


def system_markers(target_lang: str, prompt_shots: Sequence[int]) -> dict[str, str]:
    """Get the fixed marker shape of every system in the results.

    Args:
        target_lang (str): The assigned target language.
        prompt_shots (Sequence[int]): The number of demonstrations of each prompting baseline.

    Returns:
        dict[str, str]: A mapping from system name to marker shape.
    """
    return system_styles(target_lang, prompt_shots, PROMPT_MARKERS, PEFT_MARKERS)


def apply_style() -> None:
    """Apply a style to matplotlib."""
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            # Sizes for a figure set at the full text width of the report.
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panels(
    rows: int, columns: int, height: float, sharex: bool = False, sharey: bool = False
) -> tuple[Figure, np.ndarray]:
    """Get a figure with a grid of panels, with the given height and shared axes.

    Args:
        rows (int): The number of panel rows.
        columns (int): The number of panel columns.
        height (float): The figure height in inches.
        sharex (bool): Whether the panels share the x axis.
        sharey (bool): Whether the panels share the y axis.

    Returns:
        tuple[Figure, np.ndarray]: A figure and a 2D array of axes, indexed by row then column.
    """
    return plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, height),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        layout="constrained",
    )


def legend(fig: Figure, handles: list, ncols: int | None = None) -> None:
    """Get a legend for a figure, below its panels.

    Args:
        fig (Figure): The figure.
        handles (list): The legend entries.
        ncols (int | None): The number of columns or None for a single column. Defaults to None.
    """
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=ncols or min(len(handles), MAX_LEGEND_COLUMNS),
    )


def selected_name(run_name: str, best: str | None) -> str:
    """Get the display name of a candidate configuration, marking the selected one.

    Args:
        run_name (str): The candidate's run name.
        best (str | None): The run name of the selected configuration, if any.

    Returns:
        str: A display name.
    """
    return f"{run_name} (selected)" if run_name == best else run_name


def mark_selected(ax: Axes, categories: list[str], best: str | None) -> None:
    """Mark the selected configuration among the category labels of a horizontal bar panel.

    Args:
        ax (Axes): The panel.
        categories (list[str]): The candidates' run names, in panel order.
        best (str | None): The run name of the selected configuration, if any.
    """
    ax.set_yticks(range(len(categories)), [selected_name(run, best) for run in categories])

    for tick, run in zip(ax.get_yticklabels(), categories, strict=True):
        if run == best:
            tick.set_fontweight("bold")


def color_keys(colors: dict[str, str]) -> list[Patch]:
    """Get legend entries for a mapping from series name to colour, for bars.

    Args:
        colors (dict[str, str]): The colour of each series.

    Returns:
        list[Patch]: A list of legend entries.
    """
    return [Patch(color=color, label=name) for name, color in colors.items()]


def marker_keys(colors: dict[str, str], markers: dict[str, str]) -> list[Line2D]:
    """Get legend entries for a mapping from series name to colour and marker, for points.

    Args:
        colors (dict[str, str]): The colour of each series.
        markers (dict[str, str]): The marker shape of each series.

    Returns:
        list[Line2D]: A list of legend entries.
    """
    return [
        Line2D([], [], color=color, marker=markers[name], linestyle="none", label=name)
        for name, color in colors.items()
    ]


def offsets(count: int) -> np.ndarray:
    """Get the offsets of a number of dodged dots or bars, in the order of the members.

    Args:
        count (int): The number of members.

    Returns:
        np.ndarray: A 1D array of offsets, in the order of the members.
    """
    return (np.arange(count) - (count - 1) / 2) * GROUP_SHARE / max(count, 1)


def dots(
    ax: Axes,
    categories: Sequence[str],
    colors: dict[str, str],
    markers: dict[str, str],
    value: Callable[[str, str], float | None],
) -> None:
    """Get a panel of dodged points: one row per category, one point per series.

    Args:
        ax (Axes): The panel.
        categories (Sequence[str]): The rows, top to bottom.
        colors (dict[str, str]): The colour of each series.
        markers (dict[str, str]): The marker shape of each series.
        value (Callable[[str, str], float | None]): The value of a series in a category.
    """
    for row, category in enumerate(categories):
        present = [(name, value(name, category)) for name in colors]
        present = [(name, point) for name, point in present if point is not None]

        for (name, point), offset in zip(present, offsets(len(present)), strict=True):
            ax.plot(point, row + offset, linestyle="none", marker=markers[name], color=colors[name])

    ax.set_yticks(range(len(categories)), categories)
    ax.set_ylim(len(categories) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def bars(
    ax: Axes,
    categories: Sequence[str],
    colors: dict[str, str],
    value: Callable[[str, str], float | None],
    horizontal: bool = False,
    interval: Callable[[str, str], list[float] | None] | None = None,
    label: str | None = None,
) -> None:
    """Get a panel of grouped bars from zero: one group per category, one bar per series, with
    optional confidence intervals as error bars and the values printed at the bar ends.

    Args:
        ax (Axes): The panel.
        categories (Sequence[str]): The groups, in order.
        colors (dict[str, str]): The colour of each series.
        value (Callable[[str, str], float | None]): The value of a series in a category.
        horizontal (bool): Whether the bars run left to right, with the groups top to bottom.
        interval (Callable[[str, str], list[float] | None] | None): The [lower, upper] confidence
            bounds of a series in a category, or None for no error bars.
        label (str | None): The format of the printed values, e.g. "{:.2f}", or None for none.
    """
    width = GROUP_SHARE / len(colors)
    # Upright values overlap once the bars get narrower than the text; sideways they always fit.
    sideways = not horizontal and len(colors) > MAX_UPRIGHT_LABELS

    for position, category in enumerate(categories):
        present = [(name, value(name, category)) for name in colors]
        present = [(name, size) for name, size in present if size is not None]

        for (name, size), offset in zip(present, offsets(len(present)), strict=True):
            bounds = interval(name, category) if interval else None
            error = [[size - bounds[0]], [bounds[1] - size]] if bounds else None
            where = position + offset
            if horizontal:
                ax.barh(where, size, width, xerr=error, color=colors[name], capsize=2)
            else:
                ax.bar(where, size, width, yerr=error, color=colors[name], capsize=2)

            if label is not None:
                # Print the value past the error bar, so the two never overlap.
                end = max(size, bounds[1]) if bounds else size
                ax.annotate(
                    label.format(size),
                    (end, where) if horizontal else (where, end),
                    xytext=(3, 0) if horizontal else (0, 2),
                    textcoords="offset points",
                    ha="left" if horizontal else "center",
                    va="center" if horizontal else "bottom",
                    rotation=90 if sideways else 0,
                    fontsize=7,
                )

    ticks, limits, grid = (
        (ax.set_yticks, ax.set_ylim, "y") if horizontal else (ax.set_xticks, ax.set_xlim, "x")
    )
    ticks(range(len(categories)), categories)
    limits(*((len(categories) - 0.5, -0.5) if horizontal else (-0.5, len(categories) - 0.5)))
    ax.grid(axis=grid, visible=False)


def save(fig: Figure, out_dir: Path, name: str, formats: list[str]) -> list[str]:
    """Save a figure.

    Args:
        fig (Figure): The figure.
        out_dir (Path): The folder to save to.
        name (str): The file name stem.
        formats (list[str]): The file format.

    Returns:
        list[str]: The paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path)
        paths.append(str(path))

    plt.close(fig)

    return paths
