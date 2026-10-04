from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.constants import ADAPTER_DIR
from src.evaluation.scoring import BaseScorer
from src.finetune.conditions import evaluation_languages
from src.finetune.lora import load_adapter
from src.models.base import Model
from src.utils import tracking
from src.utils.logger import Logger
from src.utils.runs import pinned_model_cfg, read_run_info


class AdapterScorer(BaseScorer):
    """Class that scores a fine-tuned run by letting the model answer one partition."""

    def __init__(
        self,
        run_dir: str | Path | None,
        split: str,
        max_samples: int | None,
        wandb_cfg: DictConfig,
        logger: Logger,
    ) -> None:
        """Initialise the class.

        Args:
            run_dir (str | Path | None): The run folder, or None if there is none.
            split (str): The partition name.
            max_samples (int | None): The number of lines scored per language, or None for all.
            wandb_cfg (DictConfig): The wandb config.
            logger (Logger): The logger.

        Raises:
            ValueError: If no run folder is set or the split is not dev or test.
        """
        if not run_dir:
            raise ValueError(
                "No finished run to score. Train one first, or pass its folder, e.g. "
                "finetune.score.run_dir=outputs/<lang>/finetune/bilingual/<run_name>."
            )

        run_info = read_run_info(Path(run_dir))
        self.run_cfg = run_info["config"]

        super().__init__(
            Model(pinned_model_cfg(run_info), logger),
            Path(run_dir),
            split,
            max_samples,
            wandb_cfg,
            logger,
        )

    @property
    def condition(self) -> str:
        """Get the name of the scored condition.

        Returns:
            str: A condition.
        """
        return str(self.run_cfg["condition"])

    @property
    def langs(self) -> list[str]:
        """Get the languages to score.

        Returns:
            list[str]: A list of language codes.
        """
        return evaluation_languages(self.condition, self.run_cfg["target_lang"])

    @property
    def data(self) -> tuple[str, str, int]:
        """Get where the scored lines come from.

        Returns:
            tuple[str, str, int]: A tuple with the data folder, the target language and the seed.
        """
        run = self.run_cfg

        return run["data_dir"], run["target_lang"], run["seed"]

    def _attach(self) -> None:
        """Attach the run's adapter to the loaded base model."""
        load_adapter(self.model, str(self.run_dir / ADAPTER_DIR))

    def _tracking_run(self, model_info: dict[str, Any]) -> Any:
        """Get a W&B run for the scoring, or None if W&B is switched off.

        Args:
            model_info (dict[str, Any]): The model's info.

        Returns:
            Any: A W&B run, or None if W&B is switched off.
        """
        return tracking.start_finetune(
            self.wandb_cfg, self.condition, self.run_dir.name, self.run_dir
        )
