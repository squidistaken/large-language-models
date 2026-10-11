from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.cli.prompts import ask_target_lang, ask_until, parse_int, parse_splits
from src.constants import (
    BAND_NAMES,
    BAND_SIZES,
    EMOTIONS,
    LANGUAGE_BANDS,
    LANGUAGE_NAMES,
    LANGUAGE_TO_BAND,
    SPLIT_NAMES,
)
from src.data.formatting import ConversationFormatter, Prompts
from src.data.prepare_data import PreparationSummary, XEDPreparer
from src.utils.logger import Logger

LABEL = "Prepare Data"
TITLE = "Prepare Data"


def prompt(cfg: DictConfig, console: Console, logger: Logger) -> None:
    """Prompt the user for options.

    Args:
        cfg (DictConfig): The dictionary object override.
        console (Console): The console.
        logger (Logger): The logger.
    """
    table = Table(title="Assignable languages by resource band")

    table.add_column("Band", style="cyan")
    table.add_column("Approx. annotated lines")
    table.add_column("Languages")

    for band, langs in sorted(LANGUAGE_BANDS.items()):
        table.add_row(
            f"{band} - {BAND_NAMES[band]}",
            BAND_SIZES[band],
            ", ".join(f"{lang} ({LANGUAGE_NAMES[lang]})" for lang in sorted(langs)),
        )

    console.print(table)

    target_lang = ask_target_lang(console, cfg)

    raw_dir = Prompt.ask("Raw data directory", default=cfg.prepare.raw_dir)
    out_dir = Prompt.ask("Output directory", default=cfg.prepare.out_dir)
    splits = ask_until(
        console,
        "Split ratios (train,dev,test)",
        ",".join(str(value) for value in cfg.prepare.splits),
        parse_splits,
    )

    seed = ask_until(
        console,
        "Random seed",
        str(cfg.prepare.seed),
        lambda raw: parse_int(raw, "seed"),
    )
    overwrite = Confirm.ask("Overwrite existing splits?", default=cfg.prepare.overwrite)

    cfg.prepare.raw_dir = raw_dir
    cfg.prepare.out_dir = out_dir
    cfg.prepare.splits = splits
    cfg.prepare.seed = seed
    cfg.prepare.overwrite = overwrite

    band = LANGUAGE_TO_BAND[target_lang]

    console.print(
        f"\nPreparing matched English-{LANGUAGE_NAMES[target_lang]} pairs "
        f"(band {band}, {BAND_NAMES[band]})."
    )

    logger.info(
        f"prepare_data config: target_lang={target_lang} raw_dir={raw_dir} out_dir={out_dir} "
        f"splits={splits} seed={seed} overwrite={overwrite}"
    )


def _render(console: Console, summary: PreparationSummary) -> None:
    """Render a preparation summary.

    Args:
        console (Console): The console.
        summary (PreparationSummary): The summary to render.
    """
    overview = Table(title="Preparation summary")

    overview.add_column("Quantity")
    overview.add_column("Count", justify="right", style="cyan")

    alignment = summary.alignment

    overview.add_row("Target lines read", str(alignment.target_rows))
    overview.add_row("Malformed lines skipped", str(alignment.malformed_rows))
    overview.add_row("Unlabelled lines skipped", str(alignment.unlabelled_rows))
    overview.add_row("Unresolved to an alignment id", str(alignment.unresolved_rows))
    overview.add_row("Exact duplicate pairs dropped", str(summary.duplicate_rows))
    overview.add_row("Groups spanning >1 unit", str(summary.split.leakage_groups))
    overview.add_row("Aligned pairs kept", str(alignment.resolved_units))
    overview.add_row("...with gold English labels", str(alignment.gold_source_labels))

    for hops, count in sorted(alignment.units_by_hops.items()):
        overview.add_row(f"...via {hops}-hop alignment", str(count))

    console.print(overview)

    language = LANGUAGE_NAMES.get(summary.target_lang, summary.target_lang)
    splits = Table(title=f"{language} emotion support")

    splits.add_column("Emotion", style="cyan")

    for name in SPLIT_NAMES:
        splits.add_column(name, justify="right")

    for emotion in EMOTIONS:
        splits.add_row(
            emotion,
            *[str(summary.label_support.get(name, {}).get(emotion, 0)) for name in SPLIT_NAMES],
        )

    splits.add_section()
    splits.add_row("size", *[str(summary.split.split_sizes.get(name, 0)) for name in SPLIT_NAMES])

    console.print(splits)


def run(cfg: DictConfig, logger: Logger) -> None:
    """Run the action.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.

    Raises:
        SystemExit: If the settings are invalid, the raw inputs are missing, no pair could be
            aligned, or existing splits were built with other settings.
    """

    console = Console()

    try:
        formatter = ConversationFormatter(Prompts.from_cfg(cfg.prompting))
        preparer = XEDPreparer(cfg.prepare, logger, formatter)
        summary = preparer.prepare()

    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    remedy = (
        "Choose 'yes' at the overwrite prompt, or pass [cyan]prepare.overwrite=true[/cyan], "
        "to rebuild them."
    )

    if summary.skipped and summary.mismatches:
        console.print(
            f"[yellow]Splits under [cyan]{preparer.out_dir}[/cyan] were built with different "
            "settings, so they were left unchanged:[/yellow]"
        )

        for key, (existing, requested) in summary.mismatches.items():
            console.print(
                f"  {key}: existing [cyan]{existing}[/cyan], requested [cyan]{requested}[/cyan]"
            )

        console.print(remedy)

        # The requested splits do not exist, so later steps must not run on the old ones.
        raise SystemExit("Existing splits do not match the requested settings.")

    if summary.skipped:
        console.print(
            f"Splits already exist under [cyan]{preparer.out_dir}[/cyan]; showing their "
            f"summary. {remedy}"
        )

    _render(console, summary)

    logger.info("Preparation skipped." if summary.skipped else "Preparation complete.")
