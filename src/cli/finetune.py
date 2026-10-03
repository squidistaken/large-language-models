from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from src.cli.prompts import ask_model, ask_until, parse_choice
from src.cli.score_finetune import score_and_select
from src.constants import DEV
from src.finetune.conditions import conditions
from src.utils.logger import Logger

# The trainer loads torch and transformers, so it is imported only when the action runs; the
# menu then starts without them.
if TYPE_CHECKING:
    from src.finetune.trainer import FinetuneSummary

LABEL = "Fine-tune (PEFT)"
TITLE = "Parameter-Efficient Fine-Tuning"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    options = conditions(cfg.finetune.target_lang)
    condition = ask_until(
        console,
        f"Condition ({', '.join(options)})",
        cfg.finetune.condition,
        lambda raw: parse_choice(raw, options, "condition"),
    )

    model = ask_model(console, cfg)
    run_name = Prompt.ask("Run name", default=str(cfg.finetune.run_name))

    cfg.finetune.condition = condition
    cfg.finetune.run_name = run_name

    logger.info(f"finetune config: condition={condition} model={model} run_name={run_name}")


def _render(console: Console, summary: FinetuneSummary) -> None:
    """Render a fine-tuning summary.

    Args:
        console (Console): The console.
        summary (FinetuneSummary): The summary to render.
    """
    table = Table(title="Fine-tuning summary")

    table.add_column("Quantity")
    table.add_column("Value", justify="right", style="cyan")

    best = (
        "n/a" if summary.best_checkpoint_score is None else f"{summary.best_checkpoint_score:.4f}"
    )
    peak = "n/a" if summary.peak_gpu_memory_gib is None else f"{summary.peak_gpu_memory_gib} GiB"

    table.add_row("Condition", summary.condition)
    table.add_row("Model", summary.model)
    table.add_row("Revision", summary.revision_sha)
    table.add_row("Training examples", str(summary.train_examples))
    table.add_row("Dev examples", str(summary.dev_examples))
    table.add_row("Trainable parameters", f"{summary.trainable_params:,}")
    table.add_row("Total parameters", f"{summary.total_params:,}")
    table.add_row("Epochs trained", f"{summary.epochs_trained:g}")
    table.add_row(f"Best {summary.checkpoint_metric}", best)
    table.add_row("Peak GPU memory", peak)
    table.add_row("Device", summary.device)
    table.add_row("Saved to", summary.run_dir)

    console.print(table)


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or the prepared splits are missing.
    """
    from src.finetune.trainer import PEFTTrainer

    console = Console()

    try:
        summary = PEFTTrainer(cfg.finetune, cfg.model, cfg.wandb, logger).train()

    except (FileNotFoundError, ValueError, TypeError) as e:
        raise SystemExit(str(e)) from e

    _render(console, summary)

    logger.info("Fine-tuning complete.")

    cfg.finetune.score.run_dir = summary.run_dir
    cfg.finetune.score.split = DEV

    score_and_select(cfg, console, logger)
