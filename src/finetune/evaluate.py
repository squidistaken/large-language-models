from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.constants import EVAL_SPLITS
from src.data.loading import load_examples
from src.evaluation.scoring import SCORES_DIR, score_examples, scores_stem
from src.finetune.conditions import evaluation_languages
from src.finetune.lora import load_adapter
from src.finetune.trainer import ADAPTER_DIR
from src.models.base import Model
from src.utils.files import write_json
from src.utils.logger import Logger
from src.utils.runs import read_run_info


@dataclass
class ScoreSummary:
    """The scores of one fine-tuned run on one partition."""

    run_dir: str
    condition: str
    split: str
    scores_file: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)


class AdapterScorer:
    """Class that scores a fine-tuned run by letting the model answer one partition."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The finetune score config.
            logger (Logger): The logger.

        Raises:
            ValueError: If no run folder is set or the split is not dev or test.
        """
        if not cfg.run_dir:
            raise ValueError(
                "No finished run to score. Train one first, or pass its folder, e.g. "
                "finetune.score.run_dir=outputs/finetune/bilingual/<run_name>."
            )

        if cfg.split not in EVAL_SPLITS:
            raise ValueError(
                f"Unknown split '{cfg.split}'. Choose one of: {', '.join(EVAL_SPLITS)}"
            )

        self.cfg = cfg
        self.logger = logger
        self.run_dir = Path(cfg.run_dir)
        run_info = read_run_info(self.run_dir)
        self.run_cfg = OmegaConf.create(run_info["config"])
        model_cfg = OmegaConf.create(run_info["model"])

        model_cfg.revision = model_cfg.revision_sha

        self.model = Model(model_cfg, logger)

    def score(self) -> ScoreSummary:
        """Score a fine-tuned run by letting the model answer one partition.

        Returns:
            ScoreSummary: A summary of the scores.
        """
        run = self.run_cfg
        scores_dir = self.run_dir / SCORES_DIR
        stem = scores_stem(self.cfg.split, self.cfg.max_samples)
        summary = ScoreSummary(
            run_dir=str(self.run_dir),
            condition=run.condition,
            split=self.cfg.split,
            scores_file=str(scores_dir / f"{stem}.json"),
        )

        self.model.load()
        load_adapter(self.model, str(self.run_dir / ADAPTER_DIR))

        try:
            for lang in evaluation_languages(run.condition, run.target_lang):
                examples = load_examples(
                    run.data_dir,
                    run.target_lang,
                    [lang],
                    self.cfg.split,
                    self.logger,
                    self.cfg.max_samples,
                    run.seed,
                )
                summary.scores[lang] = score_examples(
                    self.model,
                    examples,
                    [example["messages"][:-1] for example in examples],
                    lang,
                    scores_dir / f"{stem}.{lang}.predictions.jsonl",
                )

                self.logger.info(f"{lang} {self.cfg.split}: {summary.scores[lang]}")

            write_json(
                Path(summary.scores_file),
                {
                    "split": self.cfg.split,
                    "max_samples": self.cfg.max_samples,
                    "model": self.model.info(),
                    "scores": summary.scores,
                },
            )

        finally:
            self.model.unload()

        return summary
