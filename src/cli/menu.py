from typing import Protocol, runtime_checkable

from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.cli import download_data, prepare_data
from src.utils.logger import Logger

PROJECT_TITLE = "Cross-Lingual Multi-Label Emotion Classification Using Large Language Models"


@runtime_checkable
class CLIAction(Protocol):
    """Interface class every module in `src.cli` must satisfy to be a menu action."""

    @property
    def LABEL(self) -> str: ...

    @property
    def TITLE(self) -> str: ...

    def run(self, cfg: DictConfig, logger: Logger) -> None: ...

    def prompt(self, cfg: DictConfig, console: Console, logger: Logger) -> None: ...


ACTIONS: dict[str, CLIAction] = {
    "download_data": download_data,
    "prepare_data": prepare_data,
}


def _print_banner(console: Console, text: str) -> None:
    """Print a full-width, centered Panel banner.

    Args:
        console (Console): The console to render the banner through.
        text (str): The banner text.
    """
    console.print(Panel(Text(text, justify="center"), style="bold cyan"))


def run_cli(cfg: DictConfig, logger: Logger) -> None:
    """Run the CLI.

    Args:
        cfg (DictConfig): The dictionary object override.
        logger (Logger): The logger.
    """
    console = Console()

    if cfg.get("action"):
        logger.info(f"Non-interactive action requested: {cfg.action}")

        module = ACTIONS.get(cfg.action)

        if module is None:
            logger.error(f"Unknown action: {cfg.action}")

            return

        _print_banner(console, module.TITLE)
        module.run(cfg, logger)

        return

    keys = list(ACTIONS.keys())
    exit_choice = str(len(keys) + 1)

    _print_banner(console, PROJECT_TITLE)

    while True:
        console.print()

        console.print("Select an Option:")

        console.print()

        for i, key in enumerate(keys, start=1):
            console.print(f"  [green]{i}[/green]. {ACTIONS[key].LABEL}")

        console.print(f"  [green]{exit_choice}[/green]. Exit")

        console.print()

        choice = Prompt.ask(
            "Choose an Action",
            choices=[str(i) for i in range(1, len(keys) + 1)] + [exit_choice],
        )

        if choice == exit_choice:
            logger.info("Exit selected.")
            console.print()
            _print_banner(console, PROJECT_TITLE)

            break

        action = keys[int(choice) - 1]
        module = ACTIONS[action]

        logger.info(f"Selected action: {action}.")

        console.print()

        _print_banner(console, module.TITLE)

        if hasattr(module, "prompt"):
            console.print()

            module.prompt(cfg, console, logger)

        console.print()

        module.run(cfg, logger)
