from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.evaluation.scoring import ScoreSummary, check_eval_split, score_languages
from src.finetune.conditions import evaluation_languages
from src.finetune.lora import load_adapter
from src.finetune.trainer import ADAPTER_DIR
from src.models.base import Model
from src.utils import tracking
from src.utils.logger import Logger
from src.utils.runs import pinned_model_cfg, read_run_info


class AdapterScorer:
    """Class that scores a fine-tuned run by letting the model answer one partition."""

    def __init__(self, cfg: DictConfig, wandb_cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune score config.
            wandb_cfg (DictConfig): The wandb config.
            logger (Logger): The logger.

        Raises:
            ValueError: If no run folder is set or the split is not dev or test.
        """
        if not cfg.run_dir:
            raise ValueError(
                "No finished run to score. Train one first, or pass its folder, e.g. "
                "finetune.score.run_dir=outputs/<lang>/finetune/bilingual/<run_name>."
            )

        check_eval_split(cfg.split)

        self.cfg = cfg
        self.wandb_cfg = wandb_cfg
        self.logger = logger
        self.run_dir = Path(cfg.run_dir)
        run_info = read_run_info(self.run_dir)
        self.run_cfg = OmegaConf.create(run_info["config"])
        self.model = Model(pinned_model_cfg(run_info), logger)

    def score(self) -> ScoreSummary:
        """Score a fine-tuned run by letting the model answer one partition.

        Returns:
            ScoreSummary: A summary of the scores.
        """
        run = self.run_cfg

        self.model.load()
        load_adapter(self.model, str(self.run_dir / ADAPTER_DIR))

        try:
            summary = score_languages(
                self.model,
                run.condition,
                run.data_dir,
                run.target_lang,
                evaluation_languages(run.condition, run.target_lang),
                self.cfg.split,
                self.cfg.max_samples,
                run.seed,
                self.run_dir,
                self.logger,
            )

        finally:
            self.model.unload()

        wandb_run = tracking.start_finetune(
            self.wandb_cfg, run.condition, self.run_dir.name, self.run_dir
        )

        tracking.log_scores(wandb_run, summary.split, summary.scores, summary.predictions)
        tracking.finish(wandb_run)

        return summary
