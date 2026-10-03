from src.constants import BILINGUAL, SOURCE_LANG, ZERO_SHOT


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
    """Get the display name of a PEFT condition.

    Args:
        condition (str): The PEFT condition.

    Returns:
        str: A display name.
    """
    return "zero-shot" if condition == ZERO_SHOT else f"PEFT-{condition}"


def systems(target_lang: str) -> list[str]:
    """Get the display names of all systems for a target language.

    Args:
        target_lang (str): The assigned target language.

    Returns:
        list[str]: A list of system names.
    """
    return [system_name(condition) for condition in (ZERO_SHOT, *conditions(target_lang))]
