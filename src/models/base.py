import gc
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.utils.hardware import device_summary, vram_gb
from src.utils.logger import Logger

DEFAULT_DTYPE = "bfloat16"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 32
QUANTIZATION_4BIT = "4bit"
DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _mapping(cfg: DictConfig, key: str) -> dict[str, Any]:
    """Get a mapping from the config, resolving any interpolations and converting keys to strings.

    Args:
        cfg (DictConfig): The config.
        key (str): The key of the mapping.

    Returns:
        dict[str, Any]: A mapping, or an empty mapping if the key is not present.
    """
    node = cfg.get(key)

    if node is None:
        return {}

    value = OmegaConf.to_container(node, resolve=True)

    return {str(name): item for name, item in value.items()} if isinstance(value, dict) else {}


class Model:
    """Class to wrap a Hugging Face causal language model."""

    auto_class = AutoModelForCausalLM

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The model config.
            logger (Logger): The logger.
        """
        self.logger = logger
        self.name = str(cfg.name)
        self.hf_id = str(cfg.hf_id)
        self.revision = str(cfg.get("revision", "main"))
        self.dtype = str(cfg.get("dtype", DEFAULT_DTYPE))
        self.quantization = cfg.get("quantization", None)
        self.batch_size = int(cfg.get("batch_size", DEFAULT_BATCH_SIZE))
        self.max_new_tokens = int(cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        self.decoding = _mapping(cfg, "decoding")
        self.chat_template_kwargs = _mapping(cfg, "chat_template_kwargs")

        if self.dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {self.dtype!r}, expected one of {sorted(DTYPES)}")

        self._model: PreTrainedModel | PeftModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._revision_sha = self.revision

    @property
    def model(self) -> PreTrainedModel | PeftModel:
        """Get the loaded model.

        Returns:
            PreTrainedModel | PeftModel: A model.

        Raises:
            RuntimeError: If the weights have not been loaded.
        """
        if self._model is None:
            raise RuntimeError(f"{self.name} is not loaded, call load() first")

        return self._model

    @model.setter
    def model(self, model: PreTrainedModel | PeftModel) -> None:
        """Replace the loaded model, e.g. with one that has a LoRA adapter on top.

        Args:
            model (PreTrainedModel | PeftModel): The new model.
        """
        self._model = model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the loaded tokenizer.

        Returns:
            PreTrainedTokenizerBase: A tokenizer.

        Raises:
            RuntimeError: If the tokenizer has not been loaded.
        """
        if self._tokenizer is None:
            raise RuntimeError(f"{self.name} is not loaded, call load() first")

        return self._tokenizer

    @property
    def is_loaded(self) -> bool:
        """Get whether the model and tokenizer are loaded.

        Returns:
            bool: True if the model and tokenizer are loaded.
        """
        return self._model is not None and self._tokenizer is not None

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the configured dtype as a torch dtype.

        Returns:
            torch.dtype: A dtype.
        """
        return DTYPES[self.dtype]

    @property
    def revision_sha(self) -> str:
        """Get the commit/configured revision the weights are resolved to.

        Returns:
            str: A commit SHA or the configured revision.
        """
        return self._revision_sha

    def quantization_config(self) -> BitsAndBytesConfig | None:
        """Build the bitsandbytes config for the configured quantization.

        Returns:
            BitsAndBytesConfig | None: A quantization config, or None for full precision.

        Raises:
            ValueError: If the quantization is neither null nor 4bit.
        """
        if self.quantization in (None, "none", "null"):
            return None

        if self.quantization != QUANTIZATION_4BIT:
            raise ValueError(
                f"Unknown quantization {self.quantization!r}, expected null or {QUANTIZATION_4BIT!r}"
            )

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.torch_dtype,
        )

    def load(self) -> None:
        """Load the tokenizer and the weights at the pinned revision."""
        if self.is_loaded:
            self.logger.warning(f"{self.name} is already loaded, skipping")

            return

        self.logger.info(
            f"Loading {self.hf_id} at revision {self.revision} onto {device_summary()}"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision)

        if tokenizer is None:
            raise RuntimeError(f"No tokenizer was found for {self.hf_id}")

        # Generation reads the newest tokens off the right edge of the batch, so the padding goes
        # on the left.
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.logger.info(f"Set the pad token to the eos token {tokenizer.eos_token!r}")

        model = self.auto_class.from_pretrained(
            self.hf_id,
            revision=self.revision,
            dtype=self.torch_dtype,
            quantization_config=self.quantization_config(),
            device_map="auto",
        )
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._revision_sha = getattr(model.config, "_commit_hash", None) or self.revision

        self.logger.info(
            f"Loaded {self.hf_id} at commit {self._revision_sha} "
            f"({self.dtype}, quantization={self.quantization}) on {model.device}"
        )

    def generate(self, conversations: list[list[dict]]) -> list[str]:
        """Generate a completion for every conversation.

        Args:
            conversations (list[list[dict]]): The prompt-side conversations.

        Returns:
            list[str]: A list holding the raw completion per conversation, in the given order.

        Raises:
            TypeError: If the loaded model has no generation head.
        """
        tokenizer = self.tokenizer
        model = self.model

        if not isinstance(model, GenerationMixin | PeftModel):
            raise TypeError(f"{self.name} was loaded as a model that cannot generate")

        batches = [
            conversations[start : start + self.batch_size]
            for start in range(0, len(conversations), self.batch_size)
        ]
        completions: list[str] = []

        for batch in self.logger.track(batches, desc="Generating"):
            prompts = [
                tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                for conversation in batch
            ]

            # The prompts are already rendered, so the tokenizer only adds what the template left
            # out.
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    **self.decoding,
                )

            # Left padding keeps every prompt the same length, so one offset drops all of them.
            new_tokens = outputs[:, inputs["input_ids"].shape[1] :]

            completions.extend(
                text.strip()
                for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )

        return completions

    def info(self) -> dict[str, Any]:
        """Describe the model and its settings, for a run manifest.

        Returns:
            dict[str, Any]: A dictionary containing the resolved run settings.
        """
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "revision": self.revision,
            "revision_sha": self.revision_sha,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "decoding": self.decoding,
            "chat_template_kwargs": self.chat_template_kwargs,
            "device": device_summary(),
            "vram_gb": round(vram_gb(), 1),
        }

    def unload(self) -> None:
        """Drop the weights and release the memory they held."""
        if not self.is_loaded:
            return

        self._model = None
        self._tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(f"Unloaded {self.name}")
