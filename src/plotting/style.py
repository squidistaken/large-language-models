from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src.finetune.conditions import system_name, systems

TEXT_WIDTH_IN = 17 / 2.54
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
SERIF = ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"]


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


def sequential_cmap() -> LinearSegmentedColormap:
    """Get the sequential colour map.

    Returns:
        LinearSegmentedColormap: A one-hue colour map from light to dark.
    """
    return LinearSegmentedColormap.from_list("xed_blue", SEQUENTIAL)


def apply_style() -> None:
    """Apply the XED style to matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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
