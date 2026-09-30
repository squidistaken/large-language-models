from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from src.cli.prompts import ask_until, parse_choice
from src.finetune.evaluate import EVAL_SPLITS, SCORES_DIR, AdapterScorer, ScoreSummary
from src.utils.logger import Logger

LABEL = "Score Fine-tuned Run"
TITLE = "Score a Fine-tuned Run"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    score_cfg = cfg.finetune.score
    default_dir = None if OmegaConf.is_missing(score_cfg, "run_dir") else score_cfg.run_dir
    run_dir = Prompt.ask("Run folder", default=default_dir)
    split = ask_until(
        console,
        f"Split ({', '.join(EVAL_SPLITS)})",
        score_cfg.split,
        lambda raw: parse_choice(raw, list(EVAL_SPLITS), "split"),
    )

    score_cfg.run_dir = run_dir
    score_cfg.split = split

    logger.info(f"score_finetune config: run_dir={run_dir} split={split}")


def _render(console: Console, summary: ScoreSummary) -> None:
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
    console.print(f"Scores and predictions saved under [cyan]{summary.run_dir}/{SCORES_DIR}[/cyan]")


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or a required file is missing.
    """
    console = Console()

    try:
        summary = AdapterScorer(cfg.finetune.score, logger).score()

    except (FileNotFoundError, ValueError, TypeError) as e:
        raise SystemExit(str(e)) from e

    _render(console, summary)

    logger.info("Scoring complete.")
