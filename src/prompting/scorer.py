from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import RUN_INFO_FILE, SOURCE_LANG, ZERO_SHOT
from src.data.formatting import ConversationFormatter
from src.evaluation.scoring import BaseScorer
from src.models.base import Model
from src.utils import tracking
from src.utils.files import write_json
from src.utils.logger import Logger
from src.prompting.demonstrations import load_demonstration_turns, with_demonstrations
from src.utils.runs import library_versions, prompting_dir

class PromptScorer(BaseScorer):
    """Class that scores a base model."""

    def __init__(
        self, cfg: DictConfig, model_cfg: DictConfig, wandb_cfg: DictConfig, logger: Logger
    ) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The prompting config.
            model_cfg (DictConfig): The model config.
            wandb_cfg (DictConfig): The wandb config.
            logger (Logger): The logger.

        Raises:
            ValueError: If the split is not dev or test.
        """
        model = Model(model_cfg, logger)
        self.cfg = cfg
        self.shots = int(cfg.get("shots") or 0)
        self.demonstration_ids = list(cfg.get("demonstration_ids") or []) if self.shots else []

        if self.shots and len(self.demonstration_ids) != self.shots:
            raise ValueError(
                f"prompting.shots={self.shots} needs {self.shots} demonstration_ids, got {len(self.demonstration_ids)}."
            )
        
        self._turns: dict[str, list[dict]] = {}

        super().__init__(
            model,
            prompting_dir(cfg.out_dir, self.condition, model.name),
            cfg.split,
            cfg.max_samples,
            wandb_cfg,
            logger,
        )

    @property
    def condition(self) -> str:
        """Get the name of the scored condition.

        Returns:
            str: A condition, e.g. zero_shot or 5_shot.
        """
        return ZERO_SHOT if self.shots == 0 else f"{self.shots}_shot"

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
    

    def _demonstration_turns(self, lang: str) -> list[dict]:
        """Get the demonstration turns of one language, loading them once.

        Args:
            lang (str): The language.

        Returns:
            list[dict]: A list of chat messages.
        """
        if lang not in self._turns:
            self._turns[lang] = load_demonstration_turns(
                self.cfg.data_dir, self.cfg.target_lang, lang, self.demonstration_ids, self.logger
            )

        return self._turns[lang]
    

    def _prepare(self, examples: list[dict], lang: str) -> list[dict]:
        """Add the demonstrations in the examples' language to every example.

        Args:
            examples (list[dict]): The conversational examples.
            lang (str): The language of the examples.

        Returns:
            list[dict]: A list of the examples to score.

        Raises:
            ValueError: If a demonstration is also a scored line.
        """
        if not self.shots:
            return examples

        leaked = {example["align_id"] for example in examples} & set(self.demonstration_ids)

        if leaked:
            raise ValueError(f"Demonstrations found in the {self.split} split: {sorted(leaked)}")

        turns = self._demonstration_turns(lang)

        return [with_demonstrations(example, turns) for example in examples]
    

    def _save(self, model_info: dict[str, Any]) -> None:
        """Save the run info of the zero-shot model.

        Args:
            model_info (dict[str, Any]): A dictionary with the model's info.
        """
        formatter = ConversationFormatter()

        demonstrations = (
            {
                "align_ids": self.demonstration_ids,
                "selection": "select_demonstrations: greedy label cover over gold, 1-hop, short train lines",
                "turns": {lang: self._demonstration_turns(lang) for lang in self.langs},
            }
            if self.shots
            else "none (zero-shot)"
        )

        write_json(
            self.run_dir / RUN_INFO_FILE,
            {
                "condition": self.condition,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "model": model_info,
                "prompt": {
                    "system": formatter.system_prompt,
                    "user_template": formatter.user_prompt("{text}"),
                    "sha256": formatter.prompt_sha256,
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
            f"{self.condition}/{self.model.name}",
            self.condition,
            self.condition,
            {"prompting": OmegaConf.to_container(self.cfg, resolve=True), "model": model_info},
            self.run_dir,
        )
