import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import peft
import torch
import transformers
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.finetune.data import load_examples, tokenise_examples
from src.finetune.model import load_model, load_tokenizer, training_dtype
from src.utils.files import write_json
from src.utils.logger import Logger

ADAPTER_DIR = "adapter"
CHECKPOINT_DIR = "checkpoints"
RUN_INFO_FILE = "run_info.json"
LOGGING_STEPS = 10


@dataclass
class FinetuneSummary:
    """Facts about one fine-tuning run, rendered by the CLI and stored in the run info."""

    condition: str
    model_name: str
    run_dir: str
    train_examples: int
    dev_examples: int
    trainable_params: int
    total_params: int
    best_dev_loss: float | None
    device: str


def describe_hardware() -> dict[str, str]:
    """Describe the machine and library versions a run used.

    Returns:
        dict[str, str]: The device name and software versions.
    """
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "Apple MPS"
    else:
        device = "CPU"

    return {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }


class PEFTTrainer:
    """Class that fine-tunes the base model with LoRA for one PEFT condition."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune config.
            logger (Logger): The logger.

        Raises:
            ValueError: If no model name is set.
        """
        if OmegaConf.is_missing(cfg, "model_name"):
            raise ValueError(
                "finetune.model_name is not set. Pass finetune.model_name=<Hugging Face model> "
                "(the primary model is still to be chosen, #19)."
            )

        self.cfg = cfg
        self.logger = logger
        self.run_dir = Path(cfg.out_dir) / cfg.condition / str(cfg.run_name)

    def train(self) -> FinetuneSummary:
        """Train the adapter, keep the best checkpoint on dev, and save it with the run info.

        Returns:
            FinetuneSummary: A collection of facts describing the run.
        """
        transformers.set_seed(self.cfg.seed)

        tokenizer = load_tokenizer(self.cfg.model_name)
        train_features = self._features("train", tokenizer, self.cfg.max_train_samples)
        dev_features = self._features("dev", tokenizer, self.cfg.max_eval_samples)
        model = load_model(self.cfg, self.logger)

        trainer = Trainer(
            model=model,
            args=self._arguments(len(train_features)),
            train_dataset=Dataset.from_list(train_features),
            eval_dataset=Dataset.from_list(dev_features),
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
            processing_class=tokenizer,
        )

        trainer.train()
        trainer.save_model(str(self.run_dir / ADAPTER_DIR))
        tokenizer.save_pretrained(str(self.run_dir / ADAPTER_DIR))

        trainable, total = model.get_nb_trainable_parameters()
        hardware = describe_hardware()
        summary = FinetuneSummary(
            condition=self.cfg.condition,
            model_name=self.cfg.model_name,
            run_dir=str(self.run_dir),
            train_examples=len(train_features),
            dev_examples=len(dev_features),
            trainable_params=trainable,
            total_params=total,
            best_dev_loss=trainer.state.best_metric,
            device=hardware["device"],
        )

        write_json(
            self.run_dir / RUN_INFO_FILE,
            {
                "summary": asdict(summary),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "hardware": hardware,
                "log_history": trainer.state.log_history,
            },
        )

        self.logger.info(f"Saved the adapter and run info to {self.run_dir}")

        return summary

    def _features(
        self, split: str, tokenizer: PreTrainedTokenizerBase, limit: int | None
    ) -> list[dict[str, list[int]]]:
        """Load and tokenise one partition, optionally keeping a random subset for quick tests.

        Args:
            split (str): The partition name.
            tokenizer (PreTrainedTokenizerBase): The model's tokenizer.
            limit (int | None): The maximum number of examples, or None for all of them.

        Returns:
            list[dict[str, list[int]]]: A list of tokenised examples.
        """
        examples = load_examples(self.cfg, split, self.logger)

        if limit is not None:
            random.Random(self.cfg.seed).shuffle(examples)
            examples = examples[:limit]

        return tokenise_examples(examples, tokenizer, self.cfg, self.logger)

    def _arguments(self, train_examples: int) -> TrainingArguments:
        """Build the training arguments from the finetune config.

        Args:
            train_examples (int): The number of training examples.

        Returns:
            TrainingArguments: The training arguments.
        """
        steps_per_epoch = math.ceil(
            train_examples / (self.cfg.batch_size * self.cfg.grad_accum_steps)
        )
        warmup_steps = math.ceil(self.cfg.warmup_ratio * steps_per_epoch * self.cfg.epochs)

        return TrainingArguments(
            output_dir=str(self.run_dir / CHECKPOINT_DIR),
            num_train_epochs=self.cfg.epochs,
            learning_rate=self.cfg.learning_rate,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=self.cfg.checkpoint_metric,
            greater_is_better=False,
            logging_steps=LOGGING_STEPS,
            seed=self.cfg.seed,
            bf16=training_dtype() == torch.bfloat16,
            report_to=self.cfg.report_to,
        )
