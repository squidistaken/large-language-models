from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import RUN_INFO_FILE, SOURCE_LANG, ZERO_SHOT
from src.data.formatting import ConversationFormatter
from src.evaluation.scoring import BaseScorer
from src.models.base import Model
from src.utils import tracking
from src.utils.files import write_json
from src.utils.logger import Logger
from src.utils.runs import library_versions, zero_shot_dir


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

        super().__init__(
            model,
            zero_shot_dir(cfg.out_dir, model.name),
            cfg.split,
            cfg.max_samples,
            wandb_cfg,
            logger,
        )

    @property
    def condition(self) -> str:
        """Get the name of the scored condition.

        Returns:
            str: A condition.
        """
        return ZERO_SHOT

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

    def _save(self, model_info: dict[str, Any]) -> None:
        """Save the run info of the zero-shot model.

        Args:
            model_info (dict[str, Any]): A dictionary with the model's info.
        """
        formatter = ConversationFormatter()

        write_json(
            self.run_dir / RUN_INFO_FILE,
            {
                "condition": ZERO_SHOT,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "model": model_info,
                "prompt": {
                    "system": formatter.system_prompt,
                    "user_template": formatter.user_prompt("{text}"),
                    "sha256": formatter.prompt_sha256,
                    "demonstrations": "none (zero-shot)",
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
            f"{ZERO_SHOT}/{self.model.name}",
            ZERO_SHOT,
            ZERO_SHOT,
            {"prompting": OmegaConf.to_container(self.cfg, resolve=True), "model": model_info},
            self.run_dir,
        )
