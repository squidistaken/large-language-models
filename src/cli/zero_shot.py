from omegaconf import DictConfig
from rich.console import Console

from src.cli.prompts import ask_model, ask_split, ask_target_lang
from src.cli.render import render_scores
from src.utils.logger import Logger

SHOTS = 0

LABEL = "Zero-shot Prompting"
TITLE = "Zero-shot Prompting of the Base Model"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    target_lang = ask_target_lang(console, cfg)
    model = ask_model(console, cfg)
    cfg.prompting.split = ask_split(console, cfg.prompting.split)

    logger.info(
        f"zero_shot config: target_lang={target_lang} model={model} split={cfg.prompting.split}"
    )


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or the prepared splits are missing.
    """
    from src.prompting.scorer import PromptScorer

    cfg.prompting.shots = 0

    try:
        summary = PromptScorer(cfg.prompting, cfg.model, cfg.wandb, logger).score()

    except (FileNotFoundError, ValueError, TypeError) as e:
        raise SystemExit(str(e)) from e

    render_scores(Console(), summary)

    logger.info("Zero-shot scoring complete.")
