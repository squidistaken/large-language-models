from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Prompt

from src.cli.render import render_comparison
from src.utils.logger import Logger

LABEL = "Compare Predictions"
TITLE = "Compare Two Systems' Predictions"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    console.print(
        "Leave a path empty to use the default: zero-shot against the selected bilingual run, "
        f"on {cfg.analysis.split} in {cfg.analysis.lang}."
    )

    baseline = Prompt.ask("Baseline predictions", default=cfg.analysis.baseline or "")
    candidate = Prompt.ask("Candidate predictions", default=cfg.analysis.candidate or "")

    cfg.analysis.baseline = baseline.strip() or None
    cfg.analysis.candidate = candidate.strip() or None

    logger.info(
        f"compare_predictions config: baseline={cfg.analysis.baseline} "
        f"candidate={cfg.analysis.candidate}"
    )


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If a prediction file is missing or the files share no line.
    """
    from src.evaluation.compare import PredictionComparer

    try:
        summary = PredictionComparer(cfg, logger).compare()

    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    render_comparison(Console(), summary)

    logger.info("Comparison complete.")
