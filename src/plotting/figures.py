from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.finetune.conditions import conditions
from src.finetune.selection import selected_run
from src.plotting.history import plot_history
from src.plotting.results import (
    APPENDIX_METRICS,
    HEADLINE_METRICS,
    plot_emotion_gain,
    plot_language_gap,
    plot_metrics,
    plot_per_emotion,
)
from src.plotting.style import apply_style, save
from src.utils.files import read_json
from src.utils.logger import Logger
from src.utils.runs import results_stem, run_dir


def make_figures(
    cfg: DictConfig,
    logger: Logger,
    run_name: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    per_emotion: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Draw the report figures: the fine-tuning history of a configuration, and the metrics and
    per-emotion F1 of its final results.

    Args:
        cfg (DictConfig): The full config.
        logger (Logger): The logger.
        run_name (str | None): The configuration; None means evaluate.run_name, else the
            dev-set selection.
        rows (list[dict[str, Any]] | None): The aggregate results; None reads them from disk.
        per_emotion (list[dict[str, Any]] | None): The per-emotion results; None reads them
            from disk.

    Returns:
        list[str]: The paths written.

    Raises:
        FileNotFoundError: If no configuration is given and none has been selected yet.
    """
    run_name = run_name or cfg.evaluate.run_name or selected_run(cfg.finetune.out_dir)

    if run_name is None:
        raise FileNotFoundError("No configuration selected yet. Pass evaluate.run_name=<name>.")

    apply_style()

    target_lang = str(cfg.finetune.target_lang)
    out_dir = Path(cfg.plot.out_dir)
    formats = [str(fmt) for fmt in cfg.plot.formats]
    stem = results_stem(cfg.evaluate.split, cfg.evaluate.max_samples, run_name)

    if rows is None or per_emotion is None:
        path = Path(cfg.evaluate.out_dir) / f"{stem}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `uv run main.py action=evaluate_final` first."
            )

        results = read_json(path)
        rows, per_emotion = results["results"], results["per_emotion"]

    runs = {
        condition: run_dir(cfg.finetune.out_dir, condition, run_name)
        for condition in conditions(target_lang)
    }
    files = [
        *save(plot_history(runs, target_lang), out_dir, f"training_history.{run_name}", formats),
        *save(
            plot_metrics(rows, target_lang, HEADLINE_METRICS, labels=True),
            out_dir,
            f"metrics.{stem}",
            formats,
        ),
        *save(
            plot_metrics(rows, target_lang, APPENDIX_METRICS, labels=False),
            out_dir,
            f"metrics_appendix.{stem}",
            formats,
        ),
        *save(
            plot_per_emotion(per_emotion, target_lang), out_dir, f"per_emotion_f1.{stem}", formats
        ),
        *save(plot_language_gap(rows, target_lang), out_dir, f"language_gap.{stem}", formats),
    ]

    if (gain := plot_emotion_gain(per_emotion, target_lang)) is not None:
        files += save(gain, out_dir, f"emotion_gain.{stem}", formats)

    logger.info(f"Wrote the figures to {out_dir}")

    return files
