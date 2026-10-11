from omegaconf import DictConfig
from rich.console import Console

from src.cli.prompts import ask_split, ask_target_lang, ask_until, parse_int
from src.cli.render import render_comparison, render_results
from src.utils.logger import Logger

LABEL = "Final Evaluation"
TITLE = "Final Evaluation of the Selected Configuration"


def parse_shots_list(raw: str) -> list[int]:
    """Parse a list of distinct numbers of shots, in the given order.

    Args:
        raw (str): The raw answer.

    Returns:
        list[int]: A list of distinct numbers of shots, in the given order.

    Raises:
        ValueError: If a value is not a whole number of at least zero.
    """
    if raw.strip().lower() in ("", "none"):
        return []

    values = [parse_int(part, "number of shots", minimum=0) for part in raw.split(",")]

    return list(dict.fromkeys(values))


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    target_lang = ask_target_lang(console, cfg)
    cfg.evaluate.split = ask_split(console, cfg.evaluate.split)
    cfg.evaluate.prompt_shots = ask_until(
        console,
        "Prompting baselines, by number of shots (comma-separated, or 'none')",
        ",".join(str(shots) for shots in cfg.evaluate.prompt_shots) or "none",
        parse_shots_list,
    )

    logger.info(
        f"evaluate_final config: target_lang={target_lang} split={cfg.evaluate.split} "
        f"prompt_shots={list(cfg.evaluate.prompt_shots)}"
    )


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid or a required file is missing.
    """
    from src.evaluation.compare import PredictionComparer
    from src.evaluation.results import FinalEvaluator
    from src.plotting.report import make_figures

    console = Console()

    try:
        summary = FinalEvaluator(cfg, logger).evaluate()
        figures = make_figures(cfg, logger, summary.run_name, summary)
        comparison = (
            PredictionComparer(cfg, logger, summary.run_name).compare()
            if 0 in cfg.evaluate.prompt_shots
            else None
        )

    except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
        raise SystemExit(str(e)) from e

    render_results(console, summary)

    for path in figures:
        console.print(f"Saved [cyan]{path}[/cyan]")

    if comparison is not None:
        render_comparison(console, comparison)

    logger.info("Final evaluation complete.")
