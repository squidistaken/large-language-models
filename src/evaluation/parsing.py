import json
from dataclasses import dataclass

from src.constants import EMOTIONS


@dataclass
class ParsedOutput:
    """The permitted labels found in one model answer, and whether the answer was malformed."""

    labels: list[str]
    malformed: bool


def parse_labels(output: str) -> ParsedOutput:
    """Parse a model answer into permitted emotion labels.

    Args:
        output (str): The raw model answer.

    Returns:
        ParsedOutput: A parsed output with the permitted labels and whether the answer was
                      malformed.
    """
    try:
        value = json.loads(output.strip())

    except json.JSONDecodeError:
        return ParsedOutput(labels=[], malformed=True)

    if not isinstance(value, list):
        return ParsedOutput(labels=[], malformed=True)

    found = {" ".join(item.split()).lower() for item in value if isinstance(item, str)}

    return ParsedOutput(
        labels=[emotion for emotion in EMOTIONS if emotion in found], malformed=False
    )
