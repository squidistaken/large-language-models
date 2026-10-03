import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.constants import EMOTIONS, SOURCE_LANG, ZERO_SHOT
from src.evaluation.metrics import AGGREGATE_METRICS, PRIMARY_METRICS, metric_label
from src.finetune.conditions import system_name
from src.plotting.style import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    TEXT_WIDTH_IN,
    sequential_cmap,
    system_colors,
)

# The headline metrics for the main text; the rest go to the appendix figure.
HEADLINE_METRICS = ("micro_f1", "jaccard_samples", "macro_f1")
APPENDIX_METRICS = tuple(key for key in AGGREGATE_METRICS if key not in HEADLINE_METRICS)
BAR_SHARE = 0.72
PANEL_COLUMNS = 3


def languages(rows: list[dict[str, Any]], target_lang: str) -> list[str]:
    """Get the evaluation languages present in the rows.

    Args:
        rows (list[dict[str, Any]]): The result rows.
        target_lang (str): The assigned target language.

    Returns:
        list[str]: A list of language codes.
    """
    present = {row["eval_lang"] for row in rows}

    return [lang for lang in (SOURCE_LANG, target_lang) if lang in present]


def _bars(
    ax: Axes,
    metric: str,
    lookup: dict[tuple[str, str], dict[str, Any]],
    colors: dict[str, str],
    langs: list[str],
    labels: bool,
) -> float:
    """Draw one metric as bars grouped by language, one bar per system.

    Args:
        ax (Axes): The panel.
        metric (str): The metric key.
        lookup (dict[tuple[str, str], dict[str, Any]]): The rows by system and language.
        colors (dict[str, str]): The colour of each system.
        langs (list[str]): The evaluation languages.
        labels (bool): Whether to print each value above its bar.

    Returns:
        float: The highest value drawn.
    """
    width = BAR_SHARE / len(colors)
    top = 0.0

    for group, lang in enumerate(langs):
        # Only systems evaluated on this language get a slot, so groups have no holes.
        present = [name for name in colors if (name, lang) in lookup]
        start = group - width * (len(present) - 1) / 2

        for index, name in enumerate(present):
            value = float(lookup[(name, lang)][metric])
            top = max(top, value)
            x = start + index * width

            ax.bar(x, value, width=width, color=colors[name], edgecolor=SURFACE, linewidth=1.5)

            if labels:
                ax.annotate(
                    f"{value:.2f}".removeprefix("0"),
                    (x, value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=INK_SECONDARY,
                )

    ax.set_title(metric_label(metric))
    ax.set_xticks(range(len(langs)), langs)
    ax.set_xlim(-0.5, len(langs) - 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(4))

    if top == 0:
        ax.text(
            0.5, 0.5, "0 for every system", transform=ax.transAxes, ha="center", color=INK_MUTED
        )

    return top


def plot_metrics(
    rows: list[dict[str, Any]], target_lang: str, metrics: tuple[str, ...], labels: bool
) -> Figure:
    """Plot metrics for every system and language, one panel per metric.

    Args:
        rows (list[dict[str, Any]]): The result rows.
        target_lang (str): The assigned target language.
        metrics (tuple[str, ...]): The metrics to plot.
        labels (bool): Whether to print the values; the panels then share one scale.

    Returns:
        Figure: A figure.
    """
    colors = {
        name: color
        for name, color in system_colors(target_lang).items()
        if any(row["system"] == name for row in rows)
    }
    langs = languages(rows, target_lang)
    lookup = {(row["system"], row["eval_lang"]): row for row in rows}
    panel_rows = math.ceil(len(metrics) / PANEL_COLUMNS)
    fig, axes = plt.subplots(
        panel_rows,
        PANEL_COLUMNS,
        figsize=(TEXT_WIDTH_IN, 1.9 * panel_rows + 0.35),
        squeeze=False,
        sharey=labels,
        layout="constrained",
    )
    tops = [
        _bars(ax, metric, lookup, colors, langs, labels)
        for ax, metric in zip(axes.flat, metrics, strict=False)
    ]

    for ax, top in zip(axes.flat, tops, strict=False):
        # Head room for the value labels; a minimum range so all-zero rates still draw.
        ax.set_ylim(0, max(max(tops) if labels else top, 0.05) * (1.18 if labels else 1.1))

    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)

    fig.legend(
        handles=[Patch(color=color, label=name) for name, color in colors.items()],
        loc="outside upper center",
        ncols=len(colors),
    )

    return fig


def plot_per_emotion(rows: list[dict[str, Any]], target_lang: str) -> Figure:
    """Plot the per-emotion F1 of every system as a heatmap, one panel per language.

    Args:
        rows (list[dict[str, Any]]): The per-emotion rows.
        target_lang (str): The assigned target language.

    Returns:
        Figure: A figure.
    """
    order = list(system_colors(target_lang))
    langs = languages(rows, target_lang)
    lookup = {(row["system"], row["eval_lang"], row["emotion"]): row for row in rows}
    systems = {
        lang: [name for name in order if (name, lang, EMOTIONS[0]) in lookup] for lang in langs
    }
    values = {
        lang: np.array(
            [[float(lookup[(name, lang, emotion)]["f1"]) for emotion in EMOTIONS] for name in names]
        )
        for lang, names in systems.items()
    }
    # One scale fitted to the data and shared by both panels, rounded out to tenths.
    low = math.floor(min(v.min() for v in values.values()) * 10) / 10
    high = math.ceil(max(v.max() for v in values.values()) * 10) / 10
    fig, axes = plt.subplots(
        len(langs),
        1,
        figsize=(TEXT_WIDTH_IN, sum(0.3 * len(names) + 0.75 for names in systems.values())),
        squeeze=False,
        layout="constrained",
        height_ratios=[len(systems[lang]) for lang in langs],
    )
    cmap = sequential_cmap()
    mesh = None

    for ax, lang in zip(axes[:, 0], langs, strict=True):
        names = systems[lang]
        support = [int(lookup[(names[0], lang, emotion)]["support"]) for emotion in EMOTIONS]
        mesh = ax.pcolormesh(
            values[lang], cmap=cmap, vmin=low, vmax=high, edgecolors=SURFACE, linewidth=1.5
        )

        for (row, col), value in np.ndenumerate(values[lang]):
            shade = (value - low) / (high - low) if high > low else 0.0

            ax.text(
                col + 0.5,
                row + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=SURFACE if shade > 0.6 else INK,
            )

        ax.set_title(f"Per-emotion F1, {lang}")
        ax.set_xticks(
            np.arange(len(EMOTIONS)) + 0.5,
            [f"{emotion}\nn={count}" for emotion, count in zip(EMOTIONS, support, strict=True)],
            fontsize=7,
        )
        ax.set_yticks(np.arange(len(names)) + 0.5, names)
        ax.invert_yaxis()
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    if mesh is not None:
        fig.colorbar(mesh, ax=axes[:, 0].tolist(), label="F1", shrink=0.8, pad=0.01, aspect=30)

    return fig


def plot_language_gap(rows: list[dict[str, Any]], target_lang: str) -> Figure:
    """Plot how much each system loses from English to the target language, per primary metric.

    Args:
        rows (list[dict[str, Any]]): The result rows.
        target_lang (str): The assigned target language.

    Returns:
        Figure: A figure.
    """
    colors = {
        name: color
        for name, color in system_colors(target_lang).items()
        if any(row["system"] == name for row in rows)
    }
    lookup = {(row["system"], row["eval_lang"]): row for row in rows}
    names = list(colors)
    fig, axes = plt.subplots(
        1,
        len(PRIMARY_METRICS),
        figsize=(TEXT_WIDTH_IN, 0.42 * len(names) + 0.9),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    for ax, metric in zip(axes[0], PRIMARY_METRICS, strict=True):
        for y, name in enumerate(names):
            source = lookup.get((name, SOURCE_LANG))
            target = lookup.get((name, target_lang))
            points = [float(row[metric]) for row in (source, target) if row is not None]

            if source is not None and target is not None:
                ax.plot(points, [y, y], color=colors[name], linewidth=2, zorder=1)

                gap = float(target[metric]) - float(source[metric])
                ax.annotate(
                    f"{gap:+.2f}",
                    (max(points), y),
                    xytext=(9, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color=INK_SECONDARY,
                )

            if source is not None:
                ax.scatter(
                    float(source[metric]),
                    y,
                    s=42,
                    facecolor=SURFACE,
                    edgecolor=colors[name],
                    linewidth=1.6,
                    zorder=2,
                )

            if target is not None:
                ax.scatter(
                    float(target[metric]),
                    y,
                    s=42,
                    color=colors[name],
                    edgecolor=SURFACE,
                    linewidth=1,
                    zorder=3,
                )

        ax.set_title(metric_label(metric))
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.margins(x=0.15)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)

    keys = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_SECONDARY,
            label=SOURCE_LANG,
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=6,
            color=INK_SECONDARY,
            label=target_lang,
        ),
    ]
    fig.legend(handles=keys, loc="outside upper center", ncols=len(keys))

    return fig


def plot_emotion_gain(rows: list[dict[str, Any]], target_lang: str) -> Figure | None:
    """Plot each PEFT system's per-emotion F1 gain over zero-shot prompting, one panel per
    language.

    Args:
        rows (list[dict[str, Any]]): The per-emotion rows.
        target_lang (str): The assigned target language.

    Returns:
        Figure | None: A figure, or None without zero-shot results to compare against.
    """
    baseline = system_name(ZERO_SHOT)
    lookup = {(row["system"], row["eval_lang"], row["emotion"]): float(row["f1"]) for row in rows}
    langs = [
        lang for lang in languages(rows, target_lang) if (baseline, lang, EMOTIONS[0]) in lookup
    ]

    if not langs:
        return None

    colors = {name: color for name, color in system_colors(target_lang).items() if name != baseline}
    fig, axes = plt.subplots(
        1,
        len(langs),
        figsize=(TEXT_WIDTH_IN, 3.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    present: set[str] = set()

    for ax, lang in zip(axes[0], langs, strict=True):
        systems = [name for name in colors if (name, lang, EMOTIONS[0]) in lookup]
        height = BAR_SHARE / len(systems)

        for row, emotion in enumerate(EMOTIONS):
            start = row - height * (len(systems) - 1) / 2

            for index, name in enumerate(systems):
                gain = lookup[(name, lang, emotion)] - lookup[(baseline, lang, emotion)]
                present.add(name)

                ax.barh(
                    start + index * height,
                    gain,
                    height=height,
                    color=colors[name],
                    edgecolor=SURFACE,
                    linewidth=1,
                )

        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_title(f"F1 gain over zero-shot, {lang}")
        ax.set_xlabel("F1 minus zero-shot F1")
        ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
        ax.set_ylim(len(EMOTIONS) - 0.5, -0.5)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)

    fig.legend(
        handles=[
            Patch(color=color, label=name) for name, color in colors.items() if name in present
        ],
        loc="outside upper center",
        ncols=len(present),
    )

    return fig
