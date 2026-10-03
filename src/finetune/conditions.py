from src.constants import SOURCE_LANG

BILINGUAL = "bilingual"


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
