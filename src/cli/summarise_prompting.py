from omegaconf import DictConfig
from rich.console import Console

from src.cli.prompts import ask_split, ask_target_lang, ask_until, parse_list
from src.cli.render import render_prompting
from src.utils.logger import Logger

LABEL = "Summarise Prompting"
TITLE = "Summarise the Prompting Runs"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    from src.prompting.summary import PromptingSummariser

    target_lang = ask_target_lang(console, cfg)
    cfg.prompting.split = ask_split(console, cfg.prompting.split)
    available = PromptingSummariser(cfg, logger).models()
    chosen = cfg.prompting.get("summary_models")

    if available:
        cfg.prompting.summary_models = ask_until(
            console,
            f"Models (comma-separated, or 'all'; with runs: {', '.join(available)})",
            ",".join(chosen) if chosen else "all",
            lambda raw: parse_list(raw, available, "models"),
        )

    logger.info(
        f"summarise_prompting config: target_lang={target_lang} split={cfg.prompting.split} "
        f"models={cfg.prompting.get('summary_models')}"
    )


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If no prompting run has scores on the partition.
    """
    from src.prompting.summary import PromptingSummariser

    try:
        summary = PromptingSummariser(cfg, logger).summarise(
            cfg.prompting.split, cfg.prompting.max_samples, cfg.prompting.get("summary_models")
        )

    except (FileNotFoundError, KeyError, ValueError) as e:
        raise SystemExit(str(e)) from e

    render_prompting(Console(), summary)

    logger.info("Prompting summary complete.")
