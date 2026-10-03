from collections.abc import Callable

from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from rich.console import Console
from rich.prompt import Prompt

from src.data.splitting import validate_splits


def ask_until[T](console: Console, question: str, default: str, parse: Callable[[str], T]) -> T:
    """Ask a question until the answer parses, reporting why each rejected answer failed.

    Args:
        console (Console): The console.
        question (str): The question to ask.
        default (str): The answer used when the user just presses return.
        parse (Callable[[str], T]): The function to parse the raw answer.

    Returns:
        T: The parsed answer.
    """
    while True:
        raw = Prompt.ask(question, default=default)

        try:
            return parse(raw)

        except ValueError as e:
            console.print(f"[red]{e}[/red]")


def ask_required(console: Console, question: str, default: str | None) -> str:
    """Ask a question until the answer is not empty.

    Args:
        console (Console): The console.
        question (str): The question to ask.
        default (str | None): The answer used when the user just presses return, if any.

    Returns:
        str: A non-empty answer.
    """
    while True:
        raw = Prompt.ask(question) if default is None else Prompt.ask(question, default=default)

        if raw.strip():
            return raw.strip()

        console.print("[red]An answer is required.[/red]")


def ask_model(console: Console, cfg: DictConfig) -> str:
    """Ask which model to use, and load the chosen one.

    Args:
        console (Console): The console.
        cfg (DictConfig): The dictionary object override.

    Returns:
        str: A chosen model.
    """
    options = GlobalHydra.instance().config_loader().get_group_options("model")
    name = ask_until(
        console,
        f"Model ({', '.join(options)})",
        str(cfg.model.name),
        lambda raw: parse_choice(raw, options, "model"),
    )

    if name != cfg.model.name:
        hydra_cfg = HydraConfig.get()
        overrides = [item for item in hydra_cfg.overrides.task if not item.startswith("model=")]
        composed = compose(hydra_cfg.job.config_name, overrides=[*overrides, f"model={name}"])
        cfg.model = composed.model

    return name


def parse_languages(raw: str, allowed: list[str]) -> list[str]:
    """Parse a comma-separated list of language codes, or `all`.

    Args:
        raw (str): The raw answer.
        allowed (list[str]): The permitted language codes.

    Returns:
        list[str]: A list of chosen language codes.

    Raises:
        ValueError: If any code is not permitted.
    """
    if raw.strip().lower() == "all":
        return list(allowed)

    languages = [lang.strip() for lang in raw.split(",") if lang.strip()]
    invalid = [lang for lang in languages if lang not in allowed]

    if invalid:
        raise ValueError(f"Unknown language code(s): {', '.join(invalid)}")

    return languages


def parse_choice(raw: str, allowed: list[str], subject: str) -> str:
    """Parse a single value that must come from a fixed set.

    Args:
        raw (str): The raw answer.
        allowed (list[str]): The permitted values.
        subject (str): The subject.

    Returns:
        str: A chosen value.

    Raises:
        ValueError: If the value is not permitted.
    """
    value = raw.strip()

    if value not in allowed:
        raise ValueError(f"Unknown {subject} '{value}'. Choose one of: {', '.join(allowed)}")

    return value


def parse_splits(raw: str) -> list[float]:
    """Parse comma-separated split ratios that must sum to one.

    Args:
        raw (str): The raw answer.

    Returns:
        list[float]: A list of parsed ratios.

    Raises:
        ValueError: If the ratios are unparseable or cannot partition the data.
    """
    try:
        splits = [float(part) for part in raw.split(",")]

    except ValueError as e:
        raise ValueError("Split ratios must be numbers.") from e

    validate_splits(splits)

    return splits


def parse_int(raw: str, subject: str) -> int:
    """Parse a whole number.

    Args:
        raw (str): The raw answer.
        subject (str): The subject.

    Returns:
        int: A parsed integer.

    Raises:
        ValueError: If the answer is not a whole number.
    """
    try:
        return int(raw.strip())

    except ValueError as e:
        raise ValueError(f"The {subject} must be a whole number, got '{raw}'.") from e
