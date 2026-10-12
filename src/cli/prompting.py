from omegaconf import DictConfig
from rich.console import Console

from src.cli.prompts import (
    ask_model,
    ask_split,
    ask_target_lang,
    ask_until,
    parse_choice,
    parse_int,
)
from src.cli.render import render_scores
from src.constants import SELECTIONS
from src.utils.logger import Logger

LABEL = "Prompting"
TITLE = "Prompting of the Base Model"


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
    cfg.prompting.shots = ask_until(
        console,
        "Shots (0 = zero-shot)",
        str(cfg.prompting.shots),
        lambda raw: parse_int(raw, "number of shots", minimum=0),
    )
    if cfg.prompting.shots and not cfg.prompting.get("demonstration_ids"):
        cfg.prompting.demonstration_selection = ask_until(
            console,
            f"Demonstration selection ({', '.join(SELECTIONS)}; random on dev only)",
            str(cfg.prompting.demonstration_selection),
            lambda raw: parse_choice(raw, list(SELECTIONS), "demonstration selection"),
        )

    templates = list(cfg.prompting.templates)
    cfg.prompting.template = ask_until(
        console,
        f"Prompt template ({', '.join(templates)}; variants on dev only)",
        str(cfg.prompting.template),
        lambda raw: parse_choice(raw, templates, "prompt template"),
    )

    logger.info(
        f"prompting config: target_lang={target_lang} model={model} "
        f"split={cfg.prompting.split} shots={cfg.prompting.shots} "
        f"template={cfg.prompting.template} "
        f"selection={cfg.prompting.get('demonstration_selection')}"
    )


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or the prepared splits are missing.
    """
    from src.models.base import Model
    from src.prompting.scorer import PromptScorer

    try:
        scorer = PromptScorer(cfg.prompting, Model(cfg.model, logger), cfg.wandb, logger)
        summary = scorer.score()

    except (FileNotFoundError, ValueError, TypeError) as e:
        raise SystemExit(str(e)) from e

    render_scores(Console(), summary)

    logger.info(f"{scorer.condition} scoring complete.")
