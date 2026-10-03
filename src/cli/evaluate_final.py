from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Confirm

from src.cli.prompts import ask_split, ask_target_lang
from src.cli.render import render_comparison, render_results
from src.utils.logger import Logger

LABEL = "Final Evaluation"
TITLE = "Final Evaluation of the Selected Configuration"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    target_lang = ask_target_lang(console, cfg)
    cfg.evaluate.split = ask_split(console, cfg.evaluate.split)
    cfg.evaluate.include_zero_shot = Confirm.ask(
        "Include the zero-shot baseline?", default=cfg.evaluate.include_zero_shot
    )

    logger.info(
        f"evaluate_final config: target_lang={target_lang} split={cfg.evaluate.split} "
        f"include_zero_shot={cfg.evaluate.include_zero_shot}"
    )


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action: score what is missing, write the results, draw the report figures, and
    compare zero-shot with the bilingual run. Already-scored runs are reused, so running it again
    only redraws the figures.

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
        figures = make_figures(cfg, logger, summary.run_name, summary.rows, summary.per_emotion)
        comparison = (
            PredictionComparer(cfg, logger, summary.run_name).compare()
            if cfg.evaluate.include_zero_shot
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
