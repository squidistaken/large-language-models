import random

from src.constants import TRAIN
from src.data.writing import SplitWriter
from src.utils.files import read_jsonl
from src.utils.logger import Logger

MAX_DEMO_CHARS = 80

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


def filter_candidates(rows: list[dict], max_chars: int = MAX_DEMO_CHARS) -> list[dict]:
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
    remaining = list(rows)

    for _ in range(k):
        gains = [len(set(row["labels"]) - covered) for row in remaining]
        best = max(gains)
        ties = [row for row, gain in zip(remaining, gains, strict=True) if gain == best]
        pick = rng.choice(ties)

        chosen.append(pick)
        covered.update(pick["labels"])
        remaining.remove(pick)

    return [row["align_id"] for row in chosen]


def show_pairs(data_dir: str, ids: list[str]) -> None:
    """Print the chosen demonstrations in English and Croatian, to check them by eye.

    Args:
        data_dir (str): The folder of the language pair, e.g. data/processed/xed/en-hr.
        ids (list[str]): The chosen alignment ids.
    """
    by_lang = {
        lang: {row["align_id"]: row for row in read_jsonl(f"{data_dir}/{lang}/train.jsonl")}
        for lang in ("en", "hr")
    }

    for align_id in ids:
        en, hr = by_lang["en"][align_id], by_lang["hr"][align_id]
        print(f"{align_id}  {en['labels']}")
        print(f"  en: {en['text']}")
        print(f"  hr: {hr['text']}")