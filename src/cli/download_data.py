from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.data.download_data import ALL_LANGUAGES, LANGUAGE_NAMES, XEDDownloader
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

    for lang in ALL_LANGUAGES:
        table.add_row(lang, LANGUAGE_NAMES[lang])

    console.print(table)

    default_langs = ",".join(cfg.data.languages)

    while True:
        raw = Prompt.ask("Languages (comma-separated, or 'all')", default=default_langs)

        if raw.strip().lower() == "all":
            languages = ALL_LANGUAGES

            break

        languages = [lang.strip() for lang in raw.split(",") if lang.strip()]
        invalid = [lang for lang in languages if lang not in ALL_LANGUAGES]

        if invalid:
            console.print(f"[red]Unknown language code(s): {', '.join(invalid)}[/red]")

            continue

        break

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

    downloader = XEDDownloader(cfg.data, logger)
    failed = downloader.download()

    if failed:
        raise SystemExit(f"Failed to download: {failed}")

    logger.info("Download complete.")
