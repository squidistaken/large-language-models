from pathlib import Path

from omegaconf import DictConfig
from rich.console import Console

from src.cli.prompts import ask_required, ask_split
from src.cli.render import render_scores, render_selection
from src.constants import DEV, TEST
from src.utils.logger import Logger
from src.utils.runs import latest_run

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

    score_cfg.run_dir = ask_required(console, "Run folder", default_dir)
    score_cfg.split = ask_split(console, score_cfg.split)

    logger.info(f"score_finetune config: run_dir={score_cfg.run_dir} split={score_cfg.split}")


def _warn_if_not_selected(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Warn when a test scoring is not of the configuration the dev set selected.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    from src.finetune.selection import selected_run

    best = selected_run(cfg.finetune.out_dir)

    if best is None or Path(cfg.finetune.score.run_dir).name != best:
        message = (
            f"Scoring {cfg.finetune.score.run_dir} on test, but it is not the configuration the "
            f"dev set selected ({best or 'no selection yet'}). Only report test scores of the "
            "selected configuration."
        )

        logger.warning(message)
        console.print(f"[yellow]{message}[/yellow]")


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

    if score_cfg.split == TEST and score_cfg.run_dir:
        _warn_if_not_selected(cfg, console, logger)

    try:
        scores = AdapterScorer(score_cfg, cfg.wandb, logger).score()

    except (FileNotFoundError, ValueError, TypeError) as e:
        raise SystemExit(str(e)) from e

    render_scores(console, scores)

    if scores.split != DEV or score_cfg.max_samples is not None:
        return

    try:
        selection = ConfigSelector(cfg.finetune, logger).select()

    except (FileNotFoundError, ValueError) as e:
        console.print(f"[yellow]Selection skipped: {e}[/yellow]")

        return

    render_selection(console, selection)

    console.print(f"Selection saved to [cyan]{cfg.finetune.out_dir}/{SELECTION_FILE}[/cyan]")


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.
    """
    score_and_select(cfg, Console(), logger)

    logger.info("Scoring complete.")
