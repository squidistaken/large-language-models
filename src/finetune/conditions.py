from collections.abc import Sequence

from src.constants import BILINGUAL, SHOT_SUFFIX, SOURCE_LANG, ZERO_SHOT
from src.utils.runs import PromptVariant


def conditions(target_lang: str) -> list[str]:
    """Get the three PEFT conditions for a target language.

    Args:
        target_lang (str): The assigned target language.

    Returns:
        list[str]: A list of condition names.
    """
    return [SOURCE_LANG, target_lang, BILINGUAL]


def condition_languages(condition: str, target_lang: str) -> list[str]:
    """Get the languages that a PEFT condition is trained on.

    Args:
        condition (str): The condition.
        target_lang (str): The assigned target language.

    Returns:
        list[str]: A list of language codes.

    Raises:
        ValueError: If the condition is unknown.
    """
    options = dict(
        zip(
            conditions(target_lang),
            [[SOURCE_LANG], [target_lang], [SOURCE_LANG, target_lang]],
            strict=True,
        )
    )

    if condition not in options:
        raise ValueError(f"Unknown condition '{condition}'. Choose one of: {', '.join(options)}")

    return options[condition]


def evaluation_languages(condition: str, target_lang: str) -> list[str]:
    """Get the languages that a PEFT condition is evaluated on.

    Args:
        condition (str): The condition.
        target_lang (str): The assigned target language.

    Returns:
        list[str]: A list of language codes.
    """
    if condition_languages(condition, target_lang) == [target_lang]:
        return [target_lang]

    return [SOURCE_LANG, target_lang]


def system_name(condition: str) -> str:
    """Get the display name of a prompting or PEFT condition.

    Args:
        condition (str): The condition.

    Returns:
        str: A display name.
    """
    shots, _, variant = condition.partition("-")

    if shots == ZERO_SHOT or shots.endswith(SHOT_SUFFIX):
        name = shots.replace("_", "-")

        return f"{name} ({variant})" if variant else name

    return f"PEFT-{condition}"


def prompt_baselines(shots: Sequence[int]) -> list[str]:
    """Get the prompting conditions the PEFT conditions are compared with.

    Args:
        shots (Sequence[int]): The number of demonstrations of each baseline.

    Returns:
        list[str]: A list of condition names.
    """
    return [PromptVariant(shots=int(count)).condition for count in shots]
