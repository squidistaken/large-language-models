from omegaconf import DictConfig
from rich.console import Console

from src.cli.prompts import ask_target_lang
from src.utils.logger import Logger

from src.constants import SOURCE_LANG

LABEL = "Select demonstrations"
TITLE = "Select Five-shot Demonstrations"

def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    target_lang = ask_target_lang(console, cfg)

    logger.info(f"select_demos config: target_lang={target_lang} k={cfg.prompting.demo_k}")


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or the prepared splits are missing.
    """
    from src.prompting.demos import (
        select_demonstrations, 
        load_train_rows, 
        filter_candidates, 
        show_pairs
    )

    p = cfg.prompting

    try:
        source = load_train_rows(p.data_dir, p.target_lang, SOURCE_LANG, logger)
        target_ids = {
            row["align_id"] for row in load_train_rows(p.data_dir, p.target_lang, p.target_lang, logger)
        }   
        ids = select_demonstrations(filter_candidates(source), k = p.demo_k, seed = p.seed)

    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    missing = [align_id for align_id in ids if align_id not in target_ids]

    if missing:
        raise SystemExit(f"Demonstrations missing from {p.target_lang} train: {missing}")

    logger.info(f"Selected demonstrations: {ids}")
    Console().print(f"demo_ids: [{', '.join(repr(i) for i in ids)}]")
