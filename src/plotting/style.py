from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.constants import SOURCE_LANG
from src.finetune.conditions import system_name, systems

TEXT_WIDTH_IN = 17 / 2.54
CATEGORICAL = ("tab:blue", "tab:orange", "tab:green", "tab:red")
NEUTRAL = "black"

UNIT = (0.0, 1.0)  # every metric and share bounded by 0 and 1
LOSS = (0.0, 1.25)  # cross-entropy per answer token
GAIN = (-0.3, 0.3)  # difference in F1
SHARE = (0.0, 0.5)  # share of lines carrying one emotion
HOURS = (0.0, 2.0)  # training time per PEFT condition
GROUP_SHARE = 0.72
Estimate = tuple[float, list[float] | None]


def system_colors(target_lang: str) -> dict[str, str]:
    """Get the fixed colour of every system in the results.

    Args:
        target_lang (str): The assigned target language.

    Returns:
        dict[str, str]: A mapping from system name to hex colour.
    """
    return dict(zip(systems(target_lang), CATEGORICAL, strict=True))


def condition_color(condition: str, target_lang: str) -> str:
    """Get the colour of a PEFT condition.

    Args:
        condition (str): The PEFT condition.
        target_lang (str): The assigned target language.

    Returns:
        str: A hex colour.
    """
    return system_colors(target_lang)[system_name(condition)]


def apply_style() -> None:
    """Apply a style to matplotlib."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
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


def legend(fig: Figure, handles: list) -> None:
    """Get a legend for a figure.

    Args:
        fig (Figure): The figure.
        handles (list): The legend entries.
    """
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))


def color_keys(colors: dict[str, str]) -> list[Patch]:
    """Get legend entries for a mapping from system name to colour.

    Args:
        colors (dict[str, str]): The colour of each series.

    Returns:
        list[Patch]: A list of legend entries.
    """
    return [Patch(color=color, label=name) for name, color in colors.items()]


def marker(lang: str, color: str = NEUTRAL) -> dict[str, Any]:
    """Get the marker style for a language: filled for the source language, hollow for others.

    Args:
        lang (str): The language.
        color (str): The colour of the series.

    Returns:
        dict[str, Any]: A dictionary of marker properties for a Line2D.
    """
    face = "white" if lang == SOURCE_LANG else color

    return {"markerfacecolor": face, "markeredgecolor": color, "markeredgewidth": 1.3}


def language_keys(langs: Sequence[str]) -> list[Line2D]:
    """Get legend entries for a list of languages, with the source language filled and others hollow.

    Args:
        langs (Sequence[str]): The languages.

    Returns:
        list[Line2D]: A list of legend entries.
    """
    return [
        Line2D([], [], linestyle="none", marker="o", markersize=6, label=lang, **marker(lang))
        for lang in langs
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
    series: dict[str, str],
    value: Callable[[str, str], Estimate | None],
    lang: Callable[[str], str],
) -> None:
    """Get a panel of dodged dots with optional confidence intervals: one row per category, one dot
    per series.

    Args:
        ax (Axes): The panel.
        categories (Sequence[str]): The rows, top to bottom.
        series (dict[str, str]): The colour of each series.
        value (Callable[[str, str], Estimate | None]): The value of a series in a category.
        lang (Callable[[str], str]): The language of a series, which sets the marker fill.
    """
    for row, category in enumerate(categories):
        present = [(name, value(name, category)) for name in series]
        present = [(name, estimate) for name, estimate in present if estimate is not None]

        for (name, (point, interval)), offset in zip(present, offsets(len(present)), strict=True):
            if interval is not None:
                ax.plot(interval, [row + offset] * 2, color=series[name], linewidth=1, alpha=0.6)

            ax.plot(point, row + offset, "o", markersize=5, **marker(lang(name), series[name]))

    ax.set_yticks(range(len(categories)), categories)
    ax.set_ylim(len(categories) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def bars(
    ax: Axes,
    categories: Sequence[str],
    series: dict[str, str],
    value: Callable[[str, str], float | None],
    horizontal: bool = False,
) -> None:
    """Get a panel of dodged bars: one row per category, one bar per series.

    Args:
        ax (Axes): The panel.
        categories (Sequence[str]): The groups, in order.
        series (dict[str, str]): The colour of each series.
        value (Callable[[str, str], float | None]): The value of a series in a category.
        horizontal (bool): Whether the bars run left to right, with the groups top to bottom.
    """
    width = GROUP_SHARE / len(series)
    draw = ax.barh if horizontal else ax.bar

    for position, category in enumerate(categories):
        present = [(name, value(name, category)) for name in series]
        present = [(name, size) for name, size in present if size is not None]

        for (name, size), offset in zip(present, offsets(len(present)), strict=True):
            draw(position + offset, size, width, color=series[name])

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
