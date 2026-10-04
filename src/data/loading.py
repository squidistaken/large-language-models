import random

from src.constants import DEFAULT_SEED
from src.data.writing import SplitWriter
from src.utils.files import read_jsonl
from src.utils.logger import Logger


def load_examples(
    data_dir: str,
    target_lang: str,
    langs: list[str],
    split: str,
    logger: Logger,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Load the conversational examples of one partition, optionally a random subset for tests.

    Args:
        data_dir (str): The base directory of the processed splits.
        target_lang (str): The assigned target language.
        langs (list[str]): The languages to load.
        split (str): The partition name.
        logger (Logger): The logger.
        limit (int | None): The maximum number of examples, or None for all of them.
        seed (int): The seed of the random subset.

    Returns:
        list[dict]: A list of conversational examples.

    Raises:
        FileNotFoundError: If a split file is missing.
    """
    writer = SplitWriter.for_pair(data_dir, target_lang, logger)
    examples = []

    for lang in langs:
        path = writer.chat_path(lang, split)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `uv run main.py action=prepare_data` first."
            )

        rows = read_jsonl(path)

        logger.info(f"Loaded {len(rows)} {split} examples from {path}")
        examples.extend(rows)

    if limit is not None:
        random.Random(seed).shuffle(examples)
        examples = examples[:limit]

    return examples
