from collections.abc import Callable

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
