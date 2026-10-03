from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.finetune.conditions import system_name
from src.plotting.style import INK_SECONDARY, SURFACE, TEXT_WIDTH_IN, condition_color
from src.utils.runs import read_run_info

# Exponential smoothing of the training loss, as in TensorBoard and W&B.
SMOOTHING = 0.8


def loss_curves(run_dir: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Read the training and dev loss of one run from its saved log history.

    Args:
        run_dir (Path): The run folder.

    Returns:
        tuple[list[tuple[float, float]], list[tuple[float, float]]]: The (epoch, loss) points of
            the training loss and of the dev loss.
    """
    history = read_run_info(run_dir)["log_history"]
    train = [(entry["epoch"], entry["loss"]) for entry in history if "loss" in entry]
    dev = [(entry["epoch"], entry["eval_loss"]) for entry in history if "eval_loss" in entry]

    return train, dev


def smooth(values: list[float]) -> list[float]:
    """Smooth a noisy series with a bias-corrected exponential moving average.

    Args:
        values (list[float]): The series.

    Returns:
        list[float]: The smoothed series.
    """
    smoothed, average = [], 0.0

    for step, value in enumerate(values, start=1):
        average = SMOOTHING * average + (1 - SMOOTHING) * value
        smoothed.append(average / (1 - SMOOTHING**step))

    return smoothed


def plot_history(run_dirs: dict[str, Path], target_lang: str) -> Figure:
    """Plot the training and dev loss of every PEFT condition.

    Args:
        run_dirs (dict[str, Path]): The run folder of each condition.
        target_lang (str): The assigned target language.

    Returns:
        Figure: A figure.
    """
    fig, axes = plt.subplots(
        1,
        len(run_dirs),
        figsize=(TEXT_WIDTH_IN, 2.3),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    for ax, (condition, run_dir) in zip(axes[0], run_dirs.items(), strict=True):
        color = condition_color(condition, target_lang)
        train, dev = loss_curves(run_dir)

        if train:
            epochs, losses = zip(*train, strict=True)

            ax.plot(epochs, losses, color=color, linewidth=0.7, alpha=0.3)
            ax.plot(epochs, smooth(list(losses)), color=color)

        if dev:
            epochs, losses = zip(*dev, strict=True)
            best = min(range(len(losses)), key=losses.__getitem__)

            ax.plot(
                epochs,
                losses,
                color=color,
                linestyle=(0, (4, 2)),
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
            )
            ax.plot(
                epochs[best],
                losses[best],
                marker="o",
                markersize=11,
                markerfacecolor="none",
                markeredgecolor=INK_SECONDARY,
                markeredgewidth=1,
            )

        ax.set_title(system_name(condition))
        ax.set_xlabel("Epoch")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)

    axes[0][0].set_ylabel("Cross-entropy loss")

    keys = [
        Line2D([], [], color=INK_SECONDARY, label="Training loss (smoothed)"),
        Line2D(
            [],
            [],
            color=INK_SECONDARY,
            linestyle=(0, (4, 2)),
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            label="Dev loss (per epoch)",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=INK_SECONDARY,
            label="Kept checkpoint",
        ),
    ]

    fig.legend(handles=keys, loc="outside upper center", ncols=len(keys))

    return fig
