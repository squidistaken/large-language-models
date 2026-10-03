from omegaconf import DictConfig, OmegaConf

from src.constants import SOURCE_LANG, ZERO_SHOT
from src.data.formatting import ConversationFormatter
from src.evaluation.scoring import ScoreSummary, check_eval_split, score_languages
from src.models.base import Model
from src.utils import tracking
from src.utils.files import write_json
from src.utils.logger import Logger
from src.utils.runs import RUN_INFO_FILE, library_versions, zero_shot_dir


class PromptScorer:
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
        check_eval_split(cfg.split)

        self.cfg = cfg
        self.wandb_cfg = wandb_cfg
        self.logger = logger
        self.model = Model(model_cfg, logger)
        self.run_dir = zero_shot_dir(cfg.out_dir, self.model.name)

    def score(self) -> ScoreSummary:
        """Let the base model answer one partition in English and the target language.

        Returns:
            ScoreSummary: A summary of the scores.
        """
        cfg = self.cfg

        self.model.load()

        try:
            summary = score_languages(
                self.model,
                ZERO_SHOT,
                cfg.data_dir,
                cfg.target_lang,
                [SOURCE_LANG, cfg.target_lang],
                cfg.split,
                cfg.max_samples,
                cfg.seed,
                self.run_dir,
                self.logger,
            )
            model_info = self.model.info()

        finally:
            self.model.unload()

        formatter = ConversationFormatter()
        config = OmegaConf.to_container(cfg, resolve=True)

        write_json(
            self.run_dir / RUN_INFO_FILE,
            {
                "condition": ZERO_SHOT,
                "config": config,
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

        wandb_run = tracking.start(
            self.wandb_cfg,
            f"{ZERO_SHOT}/{self.model.name}",
            ZERO_SHOT,
            ZERO_SHOT,
            {"prompting": config, "model": model_info},
            self.run_dir,
        )

        tracking.log_scores(wandb_run, summary.split, summary.scores, summary.predictions)
        tracking.finish(wandb_run)

        return summary
