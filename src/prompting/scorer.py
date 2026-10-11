from dataclasses import replace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import DEFAULT_TEMPLATE, RUN_INFO_FILE, SOURCE_LANG, TRAIN
from src.data.formatting import ConversationFormatter, Prompts
from src.data.writing import SplitWriter
from src.evaluation.scoring import BaseScorer
from src.models.base import Model
from src.prompting.demonstrations import choose_demonstrations, demonstration_turns
from src.utils import tracking
from src.utils.files import write_json
from src.utils.logger import Logger
from src.utils.runs import PromptVariant, library_versions, prompting_dir, read_run_info


class PromptScorer(BaseScorer):
    """Class that scores a base model."""

    def __init__(
        self, cfg: DictConfig, model: Model, wandb_cfg: DictConfig, logger: Logger
    ) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The prompting config.
            model (Model): The base model, loaded when scoring; scorers may share one.
            wandb_cfg (DictConfig): The wandb config.
            logger (Logger): The logger.

        Raises:
            FileNotFoundError: If the split manifest or a train split file is missing.
            ValueError: If the split is not dev or test, a sensitivity variant is not scored on
                        dev, the default prompt differs from the prepared one, the run folder holds
                        the model with other settings, or the demonstrations are invalid.
        """
        self.cfg = cfg
        self.variant = PromptVariant.of(cfg)
        self.variant.check_split(cfg.split)
        self.writer = SplitWriter.for_pair(cfg.data_dir, cfg.target_lang, logger)
        self.formatter = ConversationFormatter(Prompts.from_cfg(cfg), self.variant.template)

        self._check_prepared_prompt()

        self.demonstration_ids: list[str] = []
        self.demonstration_source = ""
        self._turns: dict[str, list[dict]] = {}

        if self.variant.shots:
            self._load_demonstrations(logger)

        super().__init__(
            model,
            prompting_dir(cfg.out_dir, self.condition, model.folder),
            cfg.split,
            cfg.max_samples,
            wandb_cfg,
            logger,
        )

        self._check_run_folder(model)

    @property
    def condition(self) -> str:
        """Get the name of the scored condition.

        Returns:
            str: A condition.
        """
        return self.variant.condition

    @property
    def langs(self) -> list[str]:
        """Get the languages to score.

        Returns:
            list[str]: A list of language codes.
        """
        return [SOURCE_LANG, self.cfg.target_lang]

    @property
    def data(self) -> tuple[str, str, int]:
        """Get where the scored lines come from.

        Returns:
            tuple[str, str, int]: A tuple of the data directory, the target language and the seed.
        """
        return self.cfg.data_dir, self.cfg.target_lang, self.cfg.seed

    def _check_run_folder(self, model: Model) -> None:
        """Check that the run folder holds no scores of the same model with other settings, so
        that runs at different precisions never mix.

        Args:
            model (Model): The model to score.

        Raises:
            ValueError: If the folder's earlier run used another checkpoint, dtype or quantization.
        """
        if not (self.run_dir / RUN_INFO_FILE).exists():
            return

        saved = read_run_info(self.run_dir)["model"]
        current = {"hf_id": model.hf_id, "dtype": model.dtype, "quantization": model.quantization}
        different = {
            key: saved.get(key) for key, value in current.items() if saved.get(key) != value
        }

        if different:
            raise ValueError(
                f"{self.run_dir} holds a run with {different}, not {current}. Move that folder to "
                "the name of its settings (e.g. <model>.4bit) or pick another prompting.out_dir."
            )

    def _check_prepared_prompt(self) -> None:
        """Check that the configured default prompt is the one the splits were prepared with.

        Raises:
            FileNotFoundError: If the split manifest is missing.
            ValueError: If the default prompt differs from the prepared one.
        """
        default = replace(self.formatter, template=DEFAULT_TEMPLATE)

        if self.writer.read_manifest().get("prompt_sha256") != default.prompt_sha256:
            raise ValueError(
                "prompting.templates.default, prompting.definitions or prompting.user differ from "
                f"the prompt the splits in {self.writer.out_dir} were prepared (and the PEFT runs "
                "trained) with. Add a new template for other wording, or re-run prepare_data."
            )

    def _load_demonstrations(self, logger: Logger) -> None:
        """Load the demonstrations and their turns in every language.

        Args:
            logger (Logger): The logger.

        Raises:
            FileNotFoundError: If a train split file is missing.
            ValueError: If the pinned ids do not match the shots or are not in train.
        """
        shots, selection = self.variant.shots, self.variant.selection
        pinned = list(self.cfg.get("demonstration_ids") or [])

        if pinned:
            if len(pinned) != shots:
                raise ValueError(
                    f"prompting.shots={shots} needs {shots} demonstration_ids, got {len(pinned)}."
                )

            self.demonstration_ids = pinned
            self.demonstration_source = "pinned: prompting.demonstration_ids"

        else:
            self.demonstration_ids = choose_demonstrations(
                self.writer, shots, self.cfg.seed, selection
            )
            self.demonstration_source = (
                f"selected ({selection}): one per film, over gold, 1-hop, short train lines "
                f"(seed={self.cfg.seed})"
            )

        # Train is disjoint from dev and test, so the demonstrations are never scored lines.
        for lang in self.langs:
            chat_rows = self.writer.read_chat(lang, TRAIN)
            self._turns[lang] = demonstration_turns(chat_rows, self.demonstration_ids, lang)

        logger.info(f"Demonstrations ({self.demonstration_source}): {self.demonstration_ids}")

    def _prepare(self, examples: list[dict], lang: str) -> list[dict]:
        """Prepare the conversational examples for scoring, by adding the prompt and demonstrations.

        Args:
            examples (list[dict]): The conversational examples.
            lang (str): The language of the examples.

        Returns:
            list[dict]: A list of the examples to score.
        """
        turns = self._turns.get(lang, [])

        return [self.formatter.with_prompt(example, turns) for example in examples]

    def _save(self, model_info: dict[str, Any]) -> None:
        """Save the run info of the prompted model.

        Args:
            model_info (dict[str, Any]): A dictionary with the model's info.
        """
        demonstrations = (
            {
                "align_ids": self.demonstration_ids,
                "selection": self.demonstration_source,
                "turns": self._turns,
            }
            if self.variant.shots
            else "none (zero-shot)"
        )

        write_json(
            self.run_dir / RUN_INFO_FILE,
            {
                "condition": self.condition,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "model": model_info,
                "prompt": {
                    "template": self.variant.template,
                    "system": self.formatter.system_prompt,
                    "user_template": self.formatter.user_prompt("{text}"),
                    "sha256": self.formatter.prompt_sha256,
                    "demonstrations": demonstrations,
                },
                "versions": library_versions(),
            },
        )

    def _tracking_run(self, model_info: dict[str, Any]) -> Any:
        """Get a W&B run for the scoring, or None if W&B is switched off.

        Args:
            model_info (dict[str, Any]): The model's info.

        Returns:
            Any: A W&B run, or None if W&B is switched off.
        """
        return tracking.start(
            self.wandb_cfg,
            f"{self.condition}/{self.run_dir.name}",
            self.condition,
            self.condition,
            {"prompting": OmegaConf.to_container(self.cfg, resolve=True), "model": model_info},
            self.run_dir,
        )
