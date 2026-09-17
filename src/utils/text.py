import re
import unicodedata

PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
QUOTE_TRANSLATION = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'}


def normalise(text: str) -> str:
    """Reduce a line of text to a comparison key.

    Tokenisation, punctuation and casing differ between the XED files being joined, so keys
    are only ever used for matching and are never written out.

    Args:
        text (str): The raw text.

    Returns:
        str: A normalised comparison key.
    """
    text = unicodedata.normalize("NFKC", text).translate(QUOTE_TRANSLATION)
    text = PUNCTUATION_PATTERN.sub(" ", text)

    return WHITESPACE_PATTERN.sub(" ", text).strip().lower()
