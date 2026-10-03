from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from src.cli.prompts import ask_required, ask_until, parse_choice
from src.constants import DEV, EVAL_SPLITS
from src.utils.logger import Logger
from src.utils.runs import latest_run

if TYPE_CHECKING:
    from src.finetune.evaluate import ScoreSummary
    from src.finetune.selection import SelectionSummary

LABEL = "Score Fine-tuned Run"
TITLE = "Score a Fine-tuned Run"


def _run_dir(cfg: DictConfig) -> str | None:
    """Get the run to score.

    Args:
        cfg (DictConfig): The dictionary object override.

    Returns:
        str | None: A path to the run folder, or None if no finished run is found.
    """
    if cfg.finetune.score.run_dir:
        return str(cfg.finetune.score.run_dir)

    latest = latest_run(cfg.finetune.out_dir)

    return str(latest) if latest else None


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    score_cfg = cfg.finetune.score
    default_dir = _run_dir(cfg)

    if default_dir is None:
        console.print(f"[yellow]No finished run in {cfg.finetune.out_dir} yet.[/yellow]")

    run_dir = ask_required(console, "Run folder", default_dir)
    split = ask_until(
        console,
        f"Split ({', '.join(EVAL_SPLITS)})",
        score_cfg.split,
        lambda raw: parse_choice(raw, list(EVAL_SPLITS), "split"),
    )

    score_cfg.run_dir = run_dir
    score_cfg.split = split

    logger.info(f"score_finetune config: run_dir={run_dir} split={split}")


def _render_scores(console: Console, summary: ScoreSummary) -> None:
    """Render the scores of a fine-tuned run.

    Args:
        console (Console): The console.
        summary (ScoreSummary): The summary to render.
    """
    table = Table(title=f"Scores on {summary.split} ({summary.condition} condition)")

    table.add_column("Language", style="cyan")
    table.add_column("micro-F1", justify="right")
    table.add_column("Jaccard", justify="right")
    table.add_column("Malformed", justify="right")
    table.add_column("Empty", justify="right")
    table.add_column("Lines", justify="right")

    for lang, metrics in summary.scores.items():
        table.add_row(
            lang,
            f"{metrics['micro_f1']:.4f}",
            f"{metrics['jaccard_samples']:.4f}",
            f"{metrics['malformed_rate']:.1%}",
            f"{metrics['empty_rate']:.1%}",
            str(int(metrics["count"])),
        )

    console.print(table)
    console.print(f"Scores saved to [cyan]{summary.scores_file}[/cyan]")


def _render_selection(console: Console, summary: SelectionSummary) -> None:
    """Render the ranked candidates.

    Args:
        console (Console): The console.
        summary (SelectionSummary): The summary to render.
    """
    table = Table(title=f"Candidates ranked by dev {summary.metric}")

    table.add_column("Rank", justify="right")
    table.add_column("Run name", style="cyan")

    for lang in summary.langs:
        table.add_column(f"{lang} (mean)", justify="right")

    table.add_column("Selection score", justify="right", style="bold")

    for rank, result in enumerate(summary.results, start=1):
        table.add_row(
            str(rank),
            result.run_name,
            *[f"{result.language_scores.get(lang, 0.0):.4f}" for lang in summary.langs],
            f"{result.selection_score:.4f}",
        )

    console.print(table)
    console.print(f"Rule: {summary.rule}")
    console.print(f"Final configuration: [bold cyan]{summary.best.run_name}[/bold cyan]")


def score_and_select(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Score a run, then, after a full dev scoring, select the final configuration.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or a required file is missing.
    """
    from src.finetune.evaluate import AdapterScorer
    from src.finetune.selection import SELECTION_FILE, ConfigSelector

    score_cfg = cfg.finetune.score
    score_cfg.run_dir = _run_dir(cfg)

    try:
        scores = AdapterScorer(score_cfg, logger).score()

    except (FileNotFoundError, ValueError, TypeError) as e:
        raise SystemExit(str(e)) from e

    _render_scores(console, scores)

    if scores.split != DEV or score_cfg.max_samples is not None:
        return

    try:
        selection = ConfigSelector(cfg.finetune, logger).select()

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[yellow]Selection skipped: {e}[/yellow]")

        return

    _render_selection(console, selection)

    console.print(f"Selection saved to [cyan]{cfg.finetune.out_dir}/{SELECTION_FILE}[/cyan]")


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.
    """
    score_and_select(cfg, Console(), logger)

    logger.info("Scoring complete.")
