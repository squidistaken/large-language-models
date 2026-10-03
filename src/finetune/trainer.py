import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

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

from src.constants import DEV, TRAIN
from src.data.loading import load_examples
from src.finetune.conditions import condition_languages
from src.finetune.data import tokenise_examples
from src.finetune.lora import add_lora
from src.models.base import Model
from src.utils.files import write_json
from src.utils.hardware import peak_memory_gib, reset_peak_memory
from src.utils.logger import Logger
from src.utils.runs import RUN_INFO_FILE, library_versions

ADAPTER_DIR = "adapter"
CHECKPOINT_DIR = "checkpoints"
LOGGING_STEPS = 10
CHECKPOINT_METRIC = "eval_loss"
CHECKPOINT_RULE = "Evaluate on dev after every epoch; keep the epoch with the lowest dev loss."


@dataclass
class FinetuneSummary:
    """Facts about one fine-tuning run, rendered by the CLI and stored in the run info."""

    condition: str
    model: str
    revision_sha: str
    run_dir: str
    train_examples: int
    dev_examples: int
    trainable_params: int
    total_params: int
    best_dev_loss: float | None
    peak_gpu_memory_gib: float | None
    device: str


class PEFTTrainer:
    """Class that fine-tunes the base model with LoRA for one PEFT condition."""

    def __init__(self, cfg: DictConfig, model_cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune config.
            model_cfg (DictConfig): The model config, shared with prompting.
            logger (Logger): The logger.

        Raises:
            ValueError: If the model is unknown, or the run folder exists and overwrite is off.
        """
        self.cfg = cfg
        self.logger = logger
        self.model = Model(model_cfg, logger)
        self.run_dir = Path(cfg.out_dir) / cfg.condition / str(cfg.run_name)

        if self.run_dir.exists() and not cfg.overwrite:
            raise ValueError(
                f"{self.run_dir} already exists. Choose another finetune.run_name, or pass "
                "finetune.overwrite=true to replace it (this also deletes its old scores)."
            )

    def train(self) -> FinetuneSummary:
        """Train the adapter, keep the best checkpoint on dev, and save it with the run info.

        Returns:
            FinetuneSummary: A collection of facts describing the run.
        """
        transformers.set_seed(self.cfg.seed)

        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
            self.logger.info(f"Removed the previous run in {self.run_dir}")

        cfg = self.cfg
        langs = condition_languages(cfg.condition, cfg.target_lang)
        train_examples = load_examples(
            cfg.data_dir,
            cfg.target_lang,
            langs,
            TRAIN,
            self.logger,
            cfg.max_train_samples,
            cfg.seed,
        )
        dev_examples = load_examples(
            cfg.data_dir, cfg.target_lang, langs, DEV, self.logger, cfg.max_eval_samples, cfg.seed
        )

        self.model.load()

        tokenizer = self.model.tokenizer

        # Generation pads on the left; training pads on the right so positions start at zero.
        tokenizer.padding_side = "right"

        train_features = self._features(train_examples, tokenizer)
        dev_features = self._features(dev_examples, tokenizer)
        model = add_lora(self.model, self.cfg.lora)
        trainer = Trainer(
            model=model,
            args=self._arguments(len(train_features)),
            train_dataset=Dataset.from_list(train_features),
            eval_dataset=Dataset.from_list(dev_features),
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
            processing_class=tokenizer,
        )

        reset_peak_memory()
        trainer.train()
        trainer.save_model(str(self.run_dir / ADAPTER_DIR))

        trainable, total = model.get_nb_trainable_parameters()
        model_info = self.model.info()
        summary = FinetuneSummary(
            condition=self.cfg.condition,
            model=self.model.hf_id,
            revision_sha=self.model.revision_sha,
            run_dir=str(self.run_dir),
            train_examples=len(train_features),
            dev_examples=len(dev_features),
            trainable_params=trainable,
            total_params=total,
            best_dev_loss=trainer.state.best_metric,
            peak_gpu_memory_gib=peak_memory_gib(),
            device=model_info["device"],
        )

        write_json(
            self.run_dir / RUN_INFO_FILE,
            {
                "summary": asdict(summary),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "checkpoint_rule": CHECKPOINT_RULE,
                "model": model_info,
                "versions": library_versions(),
                "log_history": trainer.state.log_history,
            },
        )

        self.logger.info(f"Saved the adapter and run info to {self.run_dir}")

        # Free the GPU for scoring.
        del trainer, model
        self.model.unload()

        return summary

    def _features(
        self, examples: list[dict], tokenizer: PreTrainedTokenizerBase
    ) -> list[dict[str, list[int]]]:
        """Tokenise examples with the model's own chat template options.

        Args:
            examples (list[dict]): The conversational examples.
            tokenizer (PreTrainedTokenizerBase): The model's tokenizer.

        Returns:
            list[dict[str, list[int]]]: A list of tokenised examples.
        """
        return tokenise_examples(
            examples, tokenizer, self.model.chat_template_kwargs, self.cfg.max_length, self.logger
        )

    def _arguments(self, train_examples: int) -> TrainingArguments:
        """Build the training arguments from the finetune config.

        Args:
            train_examples (int): The number of training examples.

        Returns:
            TrainingArguments: A collection of training arguments.
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
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            # Non-reentrant checkpointing lets gradients reach LoRA with frozen embeddings.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=CHECKPOINT_METRIC,
            greater_is_better=False,
            logging_steps=LOGGING_STEPS,
            seed=self.cfg.seed,
            full_determinism=self.cfg.full_determinism,
            bf16=self.model.torch_dtype == torch.bfloat16 and torch.cuda.is_available(),
            report_to="none" if os.getenv("WANDB_MODE", "disabled") == "disabled" else "wandb",
        )
