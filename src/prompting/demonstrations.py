import random

from src.constants import COVERAGE, SOURCE_LANG, TRAIN
from src.data.writing import SplitWriter

MAX_DEMONSTRATION_CHARS = 80
MAX_DEMONSTRATION_LABELS = 3


def demonstration_turns(chat_rows: list[dict], ids: list[str], lang: str) -> list[dict]:
    """Get the demonstrations of one language as alternating user and assistant messages.

    Args:
        chat_rows (list[dict]): The conversational train rows.
        ids (list[str]): The alignment ids of the demonstrations, in prompt order.
        lang (str): The language of the demonstrations.

    Returns:
        list[dict]: A list of chat messages, two per demonstration.

    Raises:
        ValueError: If a demonstration is not in the train split.
    """
    wanted = set(ids)
    by_id = {row["align_id"]: row for row in chat_rows if row["align_id"] in wanted}
    missing = [align_id for align_id in ids if align_id not in by_id]

    if missing:
        raise ValueError(f"Demonstrations missing from {lang} train: {missing}")

    # messages[0] is the system prompt; [1:] is the subtitle (user) and its gold answer (assistant).
    return [turn for align_id in ids for turn in by_id[align_id]["messages"][1:]]


def filter_candidates(
    rows: list[dict], target_rows: list[dict], max_chars: int = MAX_DEMONSTRATION_CHARS
) -> list[dict]:
    """Keep the rows suitable as demonstrations: gold English labels, a direct alignment and a
    short line in both languages.

    Args:
        rows (list[dict]): The canonical English train rows.
        target_rows (list[dict]): The canonical target-language train rows.
        max_chars (int): The longest subtitle allowed, in either language.

    Returns:
        list[dict]: A list of candidate English rows, sorted by alignment id.
    """
    short_targets = {row["align_id"] for row in target_rows if len(row["text"]) <= max_chars}
    candidates = [
        row
        for row in rows
        if row["label_source"] == "gold"
        and row["alignment_hops"] == 1
        and len(row["text"]) <= max_chars
        and len(row["labels"]) <= MAX_DEMONSTRATION_LABELS
        and row["align_id"] in short_targets
    ]

    return sorted(candidates, key=lambda row: row["align_id"])


def _film(align_id: str) -> str:
    """Get the film id from an alignment id.

    Args:
        align_id (str): The alignment id.

    Returns:
        str: The film id.
    """
    return align_id.split("/")[2]


def select_demonstrations(
    rows: list[dict], k: int, seed: int, selection: str = COVERAGE
) -> list[str]:
    """Select k demonstrations from the candidates, either to maximize label coverage or at random.

    Args:
        rows (list[dict]): The candidate rows.
        k (int): The number of demonstrations.
        seed (int): The seed of the random choices.
        selection (str): How to pick: coverage or random.

    Returns:
        list[str]: A list of the chosen alignment ids, in pick order.

    Raises:
        ValueError: If the candidates come from fewer than k films.
    """
    films = {_film(row["align_id"]) for row in rows}

    if len(films) < k:
        raise ValueError(
            f"Only {len(films)} films among {len(rows)} candidates; need {k} for {k} demonstrations."
        )

    rng = random.Random(seed)
    covered: set[str] = set()
    chosen: list[dict] = []
    used_films: set[str] = set()

    for _ in range(k):
        # One per film to maximize diversity
        remaining = [row for row in rows if _film(row["align_id"]) not in used_films]

        # Random selection treats every remaining row as an equal tie.
        gains = [
            len(set(row["labels"]) - covered) if selection == COVERAGE else 0 for row in remaining
        ]
        best = max(gains)
        ties = [row for row, gain in zip(remaining, gains, strict=True) if gain == best]
        pick = rng.choice(ties)

        chosen.append(pick)
        covered.update(pick["labels"])
        used_films.add(_film(pick["align_id"]))

    return [row["align_id"] for row in chosen]


def choose_demonstrations(
    writer: SplitWriter, k: int, seed: int, selection: str = COVERAGE
) -> list[str]:
    """Choose the demonstrations of a language pair from its train split.

    Args:
        writer (SplitWriter): The writer of the pair's splits.
        k (int): The number of demonstrations.
        seed (int): The seed of the random choices.
        selection (str): How to pick: coverage or random.

    Returns:
        list[str]: A list of the chosen alignment ids, in prompt order.

    Raises:
        FileNotFoundError: If a split file is missing.
        ValueError: If there are too few candidates.
    """
    source = writer.read_canonical(SOURCE_LANG, TRAIN)
    target = writer.read_canonical(writer.target_lang, TRAIN)

    return select_demonstrations(filter_candidates(source, target), k, seed, selection)
