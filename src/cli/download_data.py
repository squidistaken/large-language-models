from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.cli.prompts import ask_until, parse_languages
from src.constants import ALL_LANGUAGES, BAND_NAMES, LANGUAGE_NAMES, LANGUAGE_TO_BAND
from src.data.download_data import XEDDownloader
from src.utils.logger import Logger

LABEL = "Download Data"
TITLE = "Download Data"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    table = Table(title="Available languages")

    table.add_column("Code", style="cyan")
    table.add_column("Language")
    table.add_column("Band")

    for lang in ALL_LANGUAGES:
        band = LANGUAGE_TO_BAND.get(lang)
        label = f"{band} - {BAND_NAMES[band]}" if band else "source"

        table.add_row(lang, LANGUAGE_NAMES[lang], label)

    console.print(table)

    languages = ask_until(
        console,
        "Languages (comma-separated, or 'all')",
        ",".join(cfg.data.languages),
        lambda raw: parse_languages(raw, ALL_LANGUAGES),
    )

    out_dir = Prompt.ask("Output directory", default=cfg.data.out_dir)
    force = Confirm.ask("Force re-download?", default=cfg.data.force)

    cfg.data.languages = languages
    cfg.data.out_dir = out_dir
    cfg.data.force = force

    logger.info(f"download_data config: languages={languages} out_dir={out_dir} force={force}")


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If one or more files failed to download.
    """

    console = Console()
    downloader = XEDDownloader(cfg.data, logger)
    summary = downloader.download()

    if summary.failed:
        raise SystemExit(f"Failed to download: {summary.failed}")

    if summary.nothing_to_do:
        console.print(
            f"All {len(summary.skipped)} file(s) already exist under "
            f"[cyan]{cfg.data.out_dir}[/cyan]; nothing to do. Choose 'yes' at the re-download "
            "prompt, or pass [cyan]data.force=true[/cyan], to fetch them again."
        )

        logger.info("Download skipped.")

        return

    console.print(
        f"Downloaded {len(summary.downloaded)} file(s) to [cyan]{cfg.data.out_dir}[/cyan]"
        + (f", skipped {len(summary.skipped)} already present." if summary.skipped else ".")
    )

    logger.info("Download complete.")
