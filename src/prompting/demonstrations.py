import random

from src.constants import TRAIN
from src.data.writing import SplitWriter
from src.utils.files import read_jsonl
from src.utils.logger import Logger

MAX_DEMONSTRATION_CHARS = 80
MAX_DEMONSTRATION_LABELS = 3

def load_train_rows(data_dir: str, target_lang: str, lang: str, logger: Logger) -> list[dict]:
    """Load the canonical train rows of one language of a language pair.

    Args:
        data_dir (str): The folder holding every pair's splits.
        target_lang (str): The assigned target language.
        lang (str): The language to load.
        logger (Logger): The logger.

    Returns:
        list[dict]: A list of canonical rows.

    Raises:
        FileNotFoundError: If the split file is missing.
    """
    path = SplitWriter.for_pair(data_dir, target_lang, logger).canonical_path(lang, TRAIN)

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `uv run main.py action=prepare_data` first.")

    return read_jsonl(path)


def load_demonstration_turns(data_dir: str, target_lang: str, lang: str, ids: list[str], logger: Logger) -> list[dict]:
    """Load the demonstrations of one language as alternating user and assistant messages.

    Args:
        data_dir (str): The folder holding every pair's splits.
        target_lang (str): The assigned target language.
        lang (str): The language of the demonstrations.
        ids (list[str]): The alignment ids of the demonstrations, in prompt order.
        logger (Logger): The logger.

    Returns:
        list[dict]: A list of chat messages, two per demonstration.

    Raises:
        FileNotFoundError: If the split file is missing.
        ValueError: If a demonstration is not in the train split.
    """
    path = SplitWriter.for_pair(data_dir, target_lang, logger).chat_path(lang, TRAIN)

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `uv run main.py action=prepare_data` first.")

    by_id = {row["align_id"]: row for row in read_jsonl(path)}
    missing = [align_id for align_id in ids if align_id not in by_id]

    if missing:
        raise ValueError(f"Demonstrations missing from {lang} train: {missing}")

    # messages[0] is the system prompt; [1:] is the subtitle (user) and its gold answer (assistant).
    return [turn for align_id in ids for turn in by_id[align_id]["messages"][1:]]


def filter_candidates(rows: list[dict], max_chars: int = MAX_DEMONSTRATION_CHARS) -> list[dict]:
    """Keep the rows suitable as demonstrations: gold English labels, a direct alignment and a
    short line.

    Args:
        rows (list[dict]): The canonical English train rows.
        max_chars (int): The longest subtitle allowed.

    Returns:
        list[dict]: A list of candidate rows, sorted by alignment id.
    """
    candidates = [
        row
        for row in rows
        if row["label_source"] == "gold"
        and row["alignment_hops"] == 1
        and len(row["text"]) <= max_chars
        and len(row["labels"]) <= MAX_DEMONSTRATION_LABELS
    ]

    return sorted(candidates, key=lambda row: row["align_id"])

def select_demonstrations(rows: list[dict], k: int, seed: int) -> list[str]:
    """Greedily pick k rows that together cover as many emotions as possible, breaking ties at
    random with a fixed seed.

    Args:
        rows (list[dict]): The candidate rows.
        k (int): The number of demonstrations.
        seed (int): The seed of the tie-breaks.

    Returns:
        list[str]: A list of the chosen alignment ids, in pick order.

    Raises:
        ValueError: If there are too few candidates.
    """
    if len(rows) < k:
        raise ValueError(f"Only {len(rows)} candidates for {k} demonstrations.")

    rng = random.Random(seed)
    covered: set[str] = set()
    chosen: list[dict] = []
    used_films: set[str] = set()

    for _ in range(k):
        # One per film to maximize diversity
        remaining = [row for row in rows if row["align_id"].split("/")[2] not in used_films]

        gains = [len(set(row["labels"]) - covered) for row in remaining]
        best = max(gains)
        ties = [row for row, gain in zip(remaining, gains, strict=True) if gain == best]
        pick = rng.choice(ties)

        chosen.append(pick)
        covered.update(pick["labels"])
        used_films.add(pick["align_id"].split("/")[2])

    return [row["align_id"] for row in chosen]


def with_demonstrations(example: dict, turns: list[dict]) -> dict:
    """Insert the demonstration messages between the system message and the subtitle.

    Args:
        example (dict): A conversational example: system, user, assistant.
        messages (list[dict]): The demonstration messages.

    Returns:
        dict: A copy of the example with the demonstrations in its messages.
    """
    system, *rest = example["messages"]

    return {**example, "messages": [system, *turns, *rest]}
