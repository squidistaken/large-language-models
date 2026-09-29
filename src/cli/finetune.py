from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from src.cli.prompts import ask_until, parse_choice
from src.constants import SOURCE_LANG
from src.finetune.data import BILINGUAL
from src.finetune.trainer import FinetuneSummary, PEFTTrainer
from src.utils.logger import Logger

LABEL = "Fine-tune (PEFT)"
TITLE = "Parameter-Efficient Fine-Tuning"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    conditions = [SOURCE_LANG, cfg.finetune.target_lang, BILINGUAL]
    condition = ask_until(
        console,
        f"Condition ({', '.join(conditions)})",
        cfg.finetune.condition,
        lambda raw: parse_choice(raw, conditions, "condition"),
    )

    default_model = (
        None if OmegaConf.is_missing(cfg.finetune, "model_name") else cfg.finetune.model_name
    )
    model_name = Prompt.ask("Hugging Face model", default=default_model)
    run_name = Prompt.ask("Run name", default=str(cfg.finetune.run_name))

    cfg.finetune.condition = condition
    cfg.finetune.model_name = model_name
    cfg.finetune.run_name = run_name

    logger.info(
        f"finetune config: condition={condition} model_name={model_name} run_name={run_name}"
    )


def _render(console: Console, summary: FinetuneSummary) -> None:
    """Render a fine-tuning summary.

    Args:
        console (Console): The console.
        summary (FinetuneSummary): The summary to render.
    """
    table = Table(title="Fine-tuning summary")

    table.add_column("Quantity")
    table.add_column("Value", justify="right", style="cyan")

    best = "n/a" if summary.best_dev_loss is None else f"{summary.best_dev_loss:.4f}"

    table.add_row("Condition", summary.condition)
    table.add_row("Model", summary.model_name)
    table.add_row("Training examples", str(summary.train_examples))
    table.add_row("Dev examples", str(summary.dev_examples))
    table.add_row("Trainable parameters", f"{summary.trainable_params:,}")
    table.add_row("Total parameters", f"{summary.total_params:,}")
    table.add_row("Best dev loss", best)
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
    console = Console()

    try:
        summary = PEFTTrainer(cfg.finetune, logger).train()

    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    _render(console, summary)

    logger.info("Fine-tuning complete.")
