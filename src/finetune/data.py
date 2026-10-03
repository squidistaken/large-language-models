from typing import Any

from transformers import PreTrainedTokenizerBase

from src.utils.logger import Logger

IGNORE_INDEX = -100


def tokenise_example(
    example: dict, tokenizer: PreTrainedTokenizerBase, chat_template_kwargs: dict[str, Any]
) -> dict[str, list[int]]:
    """Tokenise one conversation so that only the assistant's answer is learned.

    Args:
        example (dict): The conversational example.
        tokenizer (PreTrainedTokenizerBase): The model's tokenizer.
        chat_template_kwargs (dict[str, Any]): The model's extra chat template options.

    Returns:
        dict[str, list[int]]: A dictionary with the input IDs, attention mask and labels for the
                              example.

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

    prompt_ids = tokenizer(prompt_text)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


def tokenise_examples(
    examples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    chat_template_kwargs: dict[str, Any],
    max_length: int,
    logger: Logger,
) -> list[dict[str, list[int]]]:
    """Tokenise a list of conversations, dropping any longer than the maximum length.

    Args:
        examples (list[dict]): The conversational examples.
        tokenizer (PreTrainedTokenizerBase): The model's tokenizer.
        chat_template_kwargs (dict[str, Any]): The model's extra chat template options.
        max_length (int): The maximum number of tokens per example.
        logger (Logger): The logger.

    Returns:
        list[dict[str, list[int]]]: A list of tokenised examples.
    """
    features = []
    too_long = 0

    for example in logger.track(examples, desc="Tokenising"):
        feature = tokenise_example(example, tokenizer, chat_template_kwargs)

        if len(feature["input_ids"]) > max_length:
            too_long += 1

            continue

        features.append(feature)

    if too_long:
        logger.warning(
            f"Dropped {too_long} examples longer than max_length={max_length}. "
            "Consider raising finetune.max_length."
        )

    return features
