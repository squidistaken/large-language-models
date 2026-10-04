import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

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

from src.constants import (
    ADAPTER_DIR,
    AGGREGATE_METRICS,
    CHECKPOINT_DIR,
    CHECKPOINT_METRICS,
    DEV,
    RUN_INFO_FILE,
    TRAIN,
)
from src.data.loading import load_examples
from src.evaluation.metrics import eval_key
from src.evaluation.scoring import score_examples
from src.finetune.conditions import condition_languages
from src.finetune.data import tokenise_examples
from src.finetune.lora import add_lora
from src.models.base import Model
from src.utils import tracking
from src.utils.files import write_json
from src.utils.hardware import peak_memory_gib, reset_peak_memory
from src.utils.logger import Logger
from src.utils.runs import RunVariant, library_versions, run_dir


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
    checkpoint_metric: str
    best_checkpoint_score: float | None
    epochs_trained: float
    peak_gpu_memory_gib: float | None
    device: str


class MetricTrainer(Trainer):
    """Class that scores the dev set every epoch, and stops early when it does not improve."""

    def __init__(
        self,
        *args: Any,
        dev_metrics: Callable[[], dict[str, float]],
        patience: int | None,
        **kwargs: Any,
    ) -> None:
        """Initialise the class.

        Args:
            *args (Any): The Trainer arguments.
            dev_metrics (Callable[[], dict[str, float]]): The function that scores dev, returning
                                                          metrics keyed eval_<name>, or nothing when
                                                          switched off.
            patience (int | None): The evaluations without improvement before stopping, or None
                                   to always train every epoch.
            **kwargs (Any): The Trainer keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.dev_metrics = dev_metrics
        self.patience = patience
        self.stale = 0

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Evaluate the dev loss, then score generated answers and check for early stopping.

        Args:
            *args (Any): The Trainer.evaluate arguments.
            **kwargs (Any): The Trainer.evaluate keyword arguments.

        Returns:
            dict[str, float]: The dev loss and metrics.
        """
        metrics = super().evaluate(*args, **kwargs)

        if scores := self.dev_metrics():
            self.log(scores)
            metrics.update(scores)

        if self.patience is not None:
            # The best score so far is updated only after this returns, so it is the previous one.
            key = str(self.args.metric_for_best_model)
            value, best = (
                metrics[eval_key(key.removeprefix("eval_"))],
                self.state.best_metric,
            )
            better = best is None or (value > best if self.args.greater_is_better else value < best)
            self.stale = 0 if better else self.stale + 1
            self.control.should_training_stop = self.stale >= self.patience

        return metrics


class PEFTTrainer:
    """Class that fine-tunes the base model with LoRA for one PEFT condition."""

    def __init__(
        self, cfg: DictConfig, model_cfg: DictConfig, wandb_cfg: DictConfig, logger: Logger
    ) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune config.
            model_cfg (DictConfig): The model config, shared with prompting.
            wandb_cfg (DictConfig): The wandb config.
            logger (Logger): The logger.

        Raises:
            ValueError: If the model is unknown, or the run folder exists and overwrite is off.
        """
        self.cfg = cfg
        self.wandb_cfg = wandb_cfg
        self.logger = logger
        self.model = Model(model_cfg, logger)
        self.run_dir = run_dir(
            cfg.out_dir, cfg.condition, RunVariant.of(cfg).folder(str(cfg.run_name))
        )

        if cfg.checkpoint_metric not in CHECKPOINT_METRICS:
            raise ValueError(
                f"Unknown finetune.checkpoint_metric '{cfg.checkpoint_metric}'. Choose one of: "
                f"{', '.join(CHECKPOINT_METRICS)}"
            )

        if cfg.checkpoint_metric != "loss" and cfg.metric_samples == 0:
            raise ValueError(
                "A metric checkpoint needs per-epoch scoring; set finetune.metric_samples."
            )

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
        train_examples = self._train_examples(langs)
        dev_examples = load_examples(
            cfg.data_dir, cfg.target_lang, langs, DEV, self.logger, cfg.max_eval_samples, cfg.seed
        )
        # The same seeded subset every epoch, so the per-epoch scores are comparable.
        metric_examples = {
            lang: load_examples(
                cfg.data_dir,
                cfg.target_lang,
                [lang],
                DEV,
                self.logger,
                cfg.metric_samples,
                cfg.seed,
            )
            for lang in (langs if cfg.metric_samples != 0 else [])
        }

        self.model.load()

        tokenizer = self.model.tokenizer

        # Generation pads on the left; training pads on the right so positions start at zero.
        tokenizer.padding_side = "right"

        train_features = self._features(train_examples, tokenizer)
        dev_features = self._features(dev_examples, tokenizer)
        model = add_lora(self.model, self.cfg.lora)

        wandb_run = tracking.start_finetune(
            self.wandb_cfg,
            cfg.condition,
            self.run_dir.name,
            self.run_dir,
            config={
                "finetune": OmegaConf.to_container(cfg, resolve=True),
                "model": self.model.info(),
                "train_examples": len(train_features),
                "dev_examples": len(dev_features),
            },
        )
        trainer = MetricTrainer(
            model=model,
            args=self._arguments(len(train_features), report_to_wandb=wandb_run is not None),
            train_dataset=Dataset.from_list(train_features),
            eval_dataset=Dataset.from_list(dev_features),
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8),
            processing_class=tokenizer,
            dev_metrics=lambda: self._dev_metrics(metric_examples),
            patience=cfg.early_stopping_patience,
        )

        reset_peak_memory()
        trainer.train()
        trainer.save_model(str(self.run_dir / ADAPTER_DIR))

        # The kept epoch is now in the adapter; its checkpoint also holds optimizer state, so drop it.
        shutil.rmtree(self.run_dir / CHECKPOINT_DIR, ignore_errors=True)

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
            checkpoint_metric=f"dev {cfg.checkpoint_metric}",
            best_checkpoint_score=trainer.state.best_metric,
            epochs_trained=trainer.state.epoch or 0.0,
            peak_gpu_memory_gib=peak_memory_gib(),
            device=model_info["device"],
        )

        write_json(
            self.run_dir / RUN_INFO_FILE,
            {
                "summary": asdict(summary),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "checkpoint_rule": self._checkpoint_rule(),
                "model": model_info,
                "versions": library_versions(),
                "log_history": trainer.state.log_history,
            },
        )

        if wandb_run is not None:
            wandb_run.summary.update(
                {f"summary/{key}": value for key, value in asdict(summary).items()}
            )

        self.logger.info(f"Saved the adapter and run info to {self.run_dir}")

        # Free the GPU for scoring.
        del trainer, model
        self.model.unload()

        return summary

    def _train_examples(self, langs: list[str]) -> list[dict]:
        """Train on the target language, plus the other languages of the condition, with a limit on
        the number of target-language examples.

        Args:
            langs (list[str]): The languages the condition trains on.

        Returns:
            list[dict]: A list of conversational examples.
        """
        cfg = self.cfg

        if cfg.max_target_samples is None or cfg.target_lang not in langs:
            return load_examples(
                cfg.data_dir,
                cfg.target_lang,
                langs,
                TRAIN,
                self.logger,
                cfg.max_train_samples,
                cfg.seed,
            )

        target = load_examples(
            cfg.data_dir,
            cfg.target_lang,
            [cfg.target_lang],
            TRAIN,
            self.logger,
            cfg.max_target_samples,
            cfg.seed,
        )
        others = [lang for lang in langs if lang != cfg.target_lang]

        if not others:
            return target

        return load_examples(cfg.data_dir, cfg.target_lang, others, TRAIN, self.logger) + target

    def _dev_metrics(self, examples: dict[str, list[dict]]) -> dict[str, float]:
        """Get the dev metrics, scored on a fixed subset of each language, or nothing when scoring
        is off.

        Args:
            examples (dict[str, list[dict]]): The dev subset of each language; empty when per-epoch
                scoring is off.

        Returns:
            dict[str, float]: The metrics, keyed eval_<lang>_<metric>, and the mean micro-F1 over
                the languages as eval_micro_f1.
        """
        metrics: dict[str, float] = {}

        for lang, lines in examples.items():
            scores, _ = score_examples(self.model, lines, lang)
            metrics.update({eval_key(key, lang): scores[key] for key in AGGREGATE_METRICS})

        if examples:
            metrics[eval_key("micro_f1")] = mean(
                metrics[eval_key("micro_f1", lang)] for lang in examples
            )

        return metrics

    def _checkpoint_rule(self) -> str:
        """Get a sentence describing how the kept checkpoint is picked and early stopping is
        applied.

        Returns:
            str: A sentence.
        """
        cfg = self.cfg
        best = "highest" if CHECKPOINT_METRICS[cfg.checkpoint_metric] else "lowest"
        subset = (
            "every dev line" if cfg.metric_samples is None else f"{cfg.metric_samples} dev lines"
        )
        rule = (
            f"Evaluate on dev {cfg.evals_per_epoch} time(s) per epoch; keep the checkpoint with the "
            f"{best} dev {cfg.checkpoint_metric}"
        )

        if cfg.checkpoint_metric != "loss":
            rule += f", scored on {subset} per language"

        if cfg.early_stopping_patience is not None:
            rule += f"; stop after {cfg.early_stopping_patience} evaluation(s) without improvement"

        return rule + "."

    def _features(
        self, examples: list[dict], tokenizer: PreTrainedTokenizerBase
    ) -> list[dict[str, list[int]]]:
        """Get the tokenised features for the Trainer, with the model's chat template and max length.

        Args:
            examples (list[dict]): The conversational examples.
            tokenizer (PreTrainedTokenizerBase): The model's tokenizer.

        Returns:
            list[dict[str, list[int]]]: A list of tokenised examples.
        """
        return tokenise_examples(
            examples, tokenizer, self.model.chat_template_kwargs, self.cfg.max_length, self.logger
        )

    def _arguments(self, train_examples: int, report_to_wandb: bool) -> TrainingArguments:
        """Get the training arguments for the Trainer, including the warmup and evaluation steps.

        Args:
            train_examples (int): The number of training examples.
            report_to_wandb (bool): Whether to stream the training logs to the open W&B run.

        Returns:
            TrainingArguments: A collection of training arguments.
        """
        steps_per_epoch = math.ceil(
            train_examples / (self.cfg.batch_size * self.cfg.grad_accum_steps)
        )
        warmup_steps = math.ceil(self.cfg.warmup_ratio * steps_per_epoch * self.cfg.epochs)
        eval_steps = max(1, math.ceil(steps_per_epoch / self.cfg.evals_per_epoch))

        return TrainingArguments(
            output_dir=str(self.run_dir / CHECKPOINT_DIR),
            num_train_epochs=self.cfg.epochs,
            learning_rate=self.cfg.learning_rate,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum_steps,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_steps=warmup_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=self.cfg.checkpoint_metric,
            greater_is_better=CHECKPOINT_METRICS[self.cfg.checkpoint_metric],
            logging_steps=self.cfg.logging_steps,
            seed=self.cfg.seed,
            full_determinism=self.cfg.full_determinism,
            bf16=self.model.torch_dtype == torch.bfloat16 and torch.cuda.is_available(),
            report_to="wandb" if report_to_wandb else "none",
        )
