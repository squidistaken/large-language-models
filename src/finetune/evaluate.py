import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import PreTrainedTokenizerBase

from src.constants import SOURCE_LANG
from src.finetune.data import chat_path, condition_languages
from src.finetune.model import NO_QUANTIZATION, load_base_model, load_tokenizer
from src.finetune.scoring import parse_labels, score
from src.finetune.trainer import ADAPTER_DIR, RUN_INFO_FILE
from src.utils.files import read_json, read_jsonl, write_json, write_jsonl
from src.utils.logger import Logger

SCORES_DIR = "scores"
EVAL_SPLITS = ("dev", "test")
DEFAULT_REVISION = "main"


@dataclass
class ScoreSummary:
    """Scores of one fine-tuned run on one partition, rendered by the CLI and saved with the run."""

    run_dir: str
    condition: str
    split: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluation_languages(condition: str, target_lang: str) -> list[str]:
    """Get the languages a PEFT condition is evaluated on, following the §3 experimental matrix.

    Args:
        condition (str): The condition: "en", the target language (e.g. "hr"), or "bilingual".
        target_lang (str): The assigned target language.

    Returns:
        list[str]: A list of language codes.
    """
    condition_languages(condition, target_lang)

    if condition == target_lang:
        return [target_lang]

    return [SOURCE_LANG, target_lang]


def inference_device() -> torch.device:
    """Choose the device to run the model on.

    Returns:
        torch.device: CUDA if available, then Apple MPS, otherwise the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class AdapterScorer:
    """Class that scores a fine-tuned run by letting the model answer one partition."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune score config.
            logger (Logger): The logger.

        Raises:
            ValueError: If no run folder is set or the split is not dev or test.
            FileNotFoundError: If the run folder has no run info.
        """
        if OmegaConf.is_missing(cfg, "run_dir"):
            raise ValueError(
                "finetune.score.run_dir is not set. Pass the folder of a trained run, e.g. "
                "finetune.score.run_dir=outputs/finetune/bilingual/<run_name>."
            )

        if cfg.split not in EVAL_SPLITS:
            raise ValueError(
                f"Unknown split '{cfg.split}'. Choose one of: {', '.join(EVAL_SPLITS)}"
            )

        self.cfg = cfg
        self.logger = logger
        self.run_dir = Path(cfg.run_dir)
        self.device = inference_device()

        info_path = self.run_dir / RUN_INFO_FILE

        if not info_path.exists():
            raise FileNotFoundError(
                f"Missing {info_path}. Is {self.run_dir} a finished finetune run?"
            )

        self.run_cfg = OmegaConf.create(read_json(info_path)["config"])

    def score(self) -> ScoreSummary:
        """Let the fine-tuned model answer every line, then parse and score the answers.

        Returns:
            ScoreSummary: The scores per evaluation language.
        """
        run = self.run_cfg
        revision = run.get("model_revision", DEFAULT_REVISION)
        tokenizer = load_tokenizer(run.model_name, revision)
        tokenizer.padding_side = "left"
        model = self._load_model(revision)
        scores_dir = self.run_dir / SCORES_DIR
        summary = ScoreSummary(
            run_dir=str(self.run_dir), condition=run.condition, split=self.cfg.split
        )

        scores_dir.mkdir(parents=True, exist_ok=True)

        for lang in evaluation_languages(run.condition, run.target_lang):
            examples = self._examples(lang)
            answers = self._generate(model, tokenizer, examples, lang)
            parsed = [parse_labels(answer) for answer in answers]
            gold = [json.loads(example["messages"][-1]["content"]) for example in examples]

            summary.scores[lang] = score(gold, parsed)

            write_jsonl(
                scores_dir / f"{self.cfg.split}.{lang}.predictions.jsonl",
                [
                    {
                        "align_id": example["align_id"],
                        "lang": lang,
                        "subtitle": example["messages"][-2]["content"],
                        "gold": labels,
                        "output": answer,
                        "predicted": output.labels,
                        "malformed": output.malformed,
                    }
                    for example, labels, answer, output in zip(
                        examples, gold, answers, parsed, strict=True
                    )
                ],
            )

            self.logger.info(f"{lang} {self.cfg.split}: {summary.scores[lang]}")

        write_json(
            scores_dir / f"{self.cfg.split}.json",
            {
                "split": self.cfg.split,
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": self.cfg.max_new_tokens,
                    "batch_size": self.cfg.batch_size,
                },
                "scores": summary.scores,
            },
        )

        return summary

    def _load_model(self, revision: str) -> PeftModel:
        """Load the base model and put the trained adapter on top.

        Args:
            revision (str): The model version (branch, tag or commit hash).

        Returns:
            PeftModel: The fine-tuned model, ready to answer.
        """
        run = self.run_cfg
        base = load_base_model(run.model_name, revision, run.quantization)
        model = PeftModel.from_pretrained(base, str(self.run_dir / ADAPTER_DIR))

        model.eval()

        if run.quantization == NO_QUANTIZATION:
            model.to(self.device)

        return model

    def _examples(self, lang: str) -> list[dict]:
        """Load one language's conversational examples, optionally a random subset for tests.

        Args:
            lang (str): The language.

        Returns:
            list[dict]: A list of conversational examples.

        Raises:
            FileNotFoundError: If the split file is missing.
        """
        path = chat_path(self.run_cfg.data_dir, self.run_cfg.target_lang, lang, self.cfg.split)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `uv run main.py action=prepare_data` first."
            )

        examples = read_jsonl(path)

        if self.cfg.max_samples is not None:
            random.Random(self.run_cfg.seed).shuffle(examples)
            examples = examples[: self.cfg.max_samples]

        return examples

    def _generate(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        examples: list[dict],
        lang: str,
    ) -> list[str]:
        """Let the model answer each example with greedy decoding.

        Args:
            model (PeftModel): The fine-tuned model.
            tokenizer (PreTrainedTokenizerBase): The model's tokenizer.
            examples (list[dict]): The conversational examples.
            lang (str): The language, for the progress bar.

        Returns:
            list[str]: The raw model answers.

        Raises:
            TypeError: If the chat template options are not a mapping.
        """
        options = OmegaConf.to_container(self.run_cfg.chat_template_kwargs, resolve=True)

        if not isinstance(options, dict):
            raise TypeError(
                "chat_template_kwargs must be a mapping, e.g. {enable_thinking: false}."
            )

        chat_template_kwargs: dict[str, Any] = {str(key): value for key, value in options.items()}
        prompts = [
            str(
                tokenizer.apply_chat_template(
                    example["messages"][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            )
            for example in examples
        ]
        answers = []
        starts = range(0, len(prompts), self.cfg.batch_size)

        for start in self.logger.track(starts, desc=f"Answering {lang} {self.cfg.split}"):
            batch = prompts[start : start + self.cfg.batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            new_ids = output_ids[:, inputs["input_ids"].shape[1] :]

            answers.extend(tokenizer.batch_decode(new_ids, skip_special_tokens=True))

        return answers
