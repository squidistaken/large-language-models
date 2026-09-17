from collections.abc import Callable

from rich.console import Console
from rich.prompt import Prompt


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
