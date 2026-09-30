from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from src.finetune.selection import SELECTION_FILE, ConfigSelector, SelectionSummary
from src.utils.logger import Logger

LABEL = "Select Final PEFT Configuration"
TITLE = "Select the Final PEFT Configuration"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    select_cfg = cfg.finetune.select
    default = (
        None
        if OmegaConf.is_missing(select_cfg, "candidates")
        else ",".join(str(name) for name in select_cfg.candidates)
    )
    raw = Prompt.ask("Candidate run names (comma-separated)", default=default)
    candidates = [name.strip() for name in (raw or "").split(",") if name.strip()]

    select_cfg.candidates = candidates

    logger.info(f"select_finetune config: candidates={candidates}")


def _render(console: Console, summary: SelectionSummary) -> None:
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


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or a candidate has not been scored.
    """
    console = Console()

    try:
        summary = ConfigSelector(cfg.finetune, logger).select()

    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    _render(console, summary)

    console.print(f"Selection saved to [cyan]{cfg.finetune.out_dir}/{SELECTION_FILE}[/cyan]")

    logger.info("Selection complete.")
