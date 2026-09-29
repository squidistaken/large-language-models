import torch
from omegaconf import DictConfig
from peft import (
    LoraConfig,
    PeftMixedModel,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from src.utils.logger import Logger

NO_QUANTIZATION = "none"
FOUR_BIT = "4bit"


def training_dtype() -> torch.dtype:
    """Choose the number format of the model weights for this machine.

    Returns:
        torch.dtype: bfloat16 on GPUs that support it, otherwise float32.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float32


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer of a Hugging Face model.

    Args:
        model_name (str): The Hugging Face model name.

    Returns:
        PreTrainedTokenizerBase: A tokenizer with a padding token.

    Raises:
        ValueError: If no tokenizer could be loaded for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f"Could not load a tokenizer for '{model_name}'.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(cfg: DictConfig, logger: Logger) -> PeftModel | PeftMixedModel:
    """Load the base model, optionally in 4-bit, and attach a LoRA adapter.

    Args:
        cfg (DictConfig): The finetune config.
        logger (Logger): The logger.

    Returns:
        PeftModel | PeftMixedModel: The base model with a trainable LoRA adapter.

    Raises:
        ValueError: If the quantization setting is unknown.
    """
    dtype = training_dtype()
    quantization_config = None

    if cfg.quantization == FOUR_BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.quantization != NO_QUANTIZATION:
        raise ValueError(
            f"Unknown quantization '{cfg.quantization}'. Choose '{NO_QUANTIZATION}' or '{FOUR_BIT}'."
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config is not None else None,
    )

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()

    logger.info(f"LoRA trains {trainable:,} of {total:,} parameters ({trainable / total:.2%}).")

    return model
