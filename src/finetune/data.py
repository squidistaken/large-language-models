from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from src.constants import SOURCE_LANG
from src.data.prepare_data import pair_name
from src.utils.files import read_jsonl
from src.utils.logger import Logger

BILINGUAL = "bilingual"

# Label value that tells the trainer to ignore a token when computing the loss.
IGNORE_INDEX = -100


def condition_languages(condition: str, target_lang: str) -> list[str]:
    """Map a PEFT condition to the languages whose data it trains on.

    Args:
        condition (str): The condition: "en", the target language (e.g. "hr"), or "bilingual".
        target_lang (str): The assigned target language.

    Returns:
        list[str]: A list of language codes.

    Raises:
        ValueError: If the condition is unknown.
    """
    options = {
        SOURCE_LANG: [SOURCE_LANG],
        target_lang: [target_lang],
        BILINGUAL: [SOURCE_LANG, target_lang],
    }

    if condition not in options:
        raise ValueError(f"Unknown condition '{condition}'. Choose one of: {', '.join(options)}")

    return options[condition]


def chat_path(data_dir: str, target_lang: str, lang: str, split: str) -> Path:
    """Get the path to one conversational split file written by prepare_data.

    Args:
        data_dir (str): The base directory of the processed splits.
        target_lang (str): The assigned target language.
        lang (str): The language of the file.
        split (str): The partition name.

    Returns:
        Path: A path.
    """
    return Path(data_dir) / pair_name(target_lang) / lang / f"{split}.chat.jsonl"


def load_examples(cfg: DictConfig, split: str, logger: Logger) -> list[dict]:
    """Load the conversational examples of one partition for the configured condition.

    Args:
        cfg (DictConfig): The finetune config.
        split (str): The partition name.
        logger (Logger): The logger.

    Returns:
        list[dict]: A list of conversational examples.

    Raises:
        FileNotFoundError: If a split file is missing.
    """
    examples = []

    for lang in condition_languages(cfg.condition, cfg.target_lang):
        path = chat_path(cfg.data_dir, cfg.target_lang, lang, split)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `uv run main.py action=prepare_data` first."
            )

        rows = read_jsonl(path)

        logger.info(f"Loaded {len(rows)} {split} examples from {path}")
        examples.extend(rows)

    return examples


def tokenise_example(
    example: dict, tokenizer: PreTrainedTokenizerBase, chat_template_kwargs: dict
) -> dict[str, list[int]]:
    """Tokenise one conversation so that only the assistant's answer is learned.

    Args:
        example (dict): The conversational example.
        tokenizer (PreTrainedTokenizerBase): The model's tokenizer.
        chat_template_kwargs (dict): Extra model-specific chat template options.

    Returns:
        dict[str, list[int]]: The token ids, attention mask and labels.

    Raises:
        ValueError: If the chat template renders the prompt inconsistently.
    """
    messages = example["messages"]

    prompt_text = str(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
    )
    full_text = str(tokenizer.apply_chat_template(messages, tokenize=False, **chat_template_kwargs))

    if not full_text.startswith(prompt_text):
        raise ValueError(
            "The chat template does not render the prompt as a prefix of the full text."
        )

    answer_text = full_text[len(prompt_text) :]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


def tokenise_examples(
    examples: list[dict], tokenizer: PreTrainedTokenizerBase, cfg: DictConfig, logger: Logger
) -> list[dict[str, list[int]]]:
    """Tokenise a list of conversations, dropping any longer than the maximum length.

    Args:
        examples (list[dict]): The conversational examples.
        tokenizer (PreTrainedTokenizerBase): The model's tokenizer.
        cfg (DictConfig): The finetune config.
        logger (Logger): The logger.

    Returns:
        list[dict[str, list[int]]]: A list of tokenised examples.
    """
    chat_template_kwargs = OmegaConf.to_container(cfg.chat_template_kwargs, resolve=True)

    if not isinstance(chat_template_kwargs, dict):
        raise TypeError(
            "finetune.chat_template_kwargs must be a mapping, e.g. {enable_thinking: false}."
        )
    features = []
    too_long = 0

    for example in logger.track(examples, desc="Tokenising"):
        feature = tokenise_example(example, tokenizer, chat_template_kwargs)

        if len(feature["input_ids"]) > cfg.max_length:
            too_long += 1

            continue

        features.append(feature)

    if too_long:
        logger.warning(
            f"Dropped {too_long} examples longer than max_length={cfg.max_length}. "
            "Consider raising finetune.max_length."
        )

    return features
