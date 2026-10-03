from pathlib import Path
from typing import Any, Literal, cast

import wandb
from omegaconf import DictConfig

from src.constants import EMOTIONS

DISABLED_MODES = {"", "disabled", "none", "off", "false"}
ENABLED_MODES = {"online", "offline"}
RUN_ID_FILE = "wandb_run_id"
FINETUNE = "finetune"
RESULTS = "results"


def start(
    cfg: DictConfig,
    name: str,
    group: str,
    job_type: str,
    config: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> Any:
    """Start or resume a W&B run.

    Args:
        cfg (DictConfig): The wandb config.
        name (str): The display name of the run.
        group (str): The group the run belongs to.
        job_type (str): The kind of job.
        config (dict[str, Any] | None): The settings to record with the run.
        run_dir (Path | None): The folder whose W&B run to resume, or None for a new run.

    Returns:
        Any: A W&B run, or None if W&B is switched off.

    Raises:
        ValueError: If the mode is not online, offline or disabled.
    """
    mode = str(cfg.get("mode") or "").strip().lower()

    if mode in DISABLED_MODES:
        return None

    if mode not in ENABLED_MODES:
        raise ValueError(f"Unknown WANDB_MODE '{cfg.mode}'. Choose online, offline or disabled.")

    id_file = run_dir / RUN_ID_FILE if run_dir is not None else None
    run_id = id_file.read_text().strip() if id_file is not None and id_file.exists() else None

    # Training and scoring in one process share the run that is still open.
    if run_id is not None and wandb.run is not None and wandb.run.id == run_id:
        return wandb.run

    run = wandb.init(
        project=cfg.project,
        entity=cfg.get("entity"),
        name=name,
        group=group,
        job_type=job_type,
        tags=[str(tag) for tag in cfg.get("tags") or []],
        config=config,
        id=run_id,
        resume="allow" if run_id else None,
        mode=cast(Literal["online", "offline"], mode),
        reinit="finish_previous",
    )

    if id_file is not None:
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text(run.id)

    return run


def start_finetune(
    cfg: DictConfig,
    condition: str,
    run_name: str,
    run_dir: Path,
    config: dict[str, Any] | None = None,
) -> Any:
    """Start or resume the W&B run of one fine-tuned adapter.

    Args:
        cfg (DictConfig): The wandb config.
        condition (str): The PEFT condition.
        run_name (str): The run name, shared by the conditions of one configuration.
        run_dir (Path): The run folder.
        config (dict[str, Any] | None): The settings to record with the run.

    Returns:
        Any: A W&B run, or None if W&B is switched off.
    """
    return start(cfg, f"{condition}/{run_name}", run_name, FINETUNE, config, run_dir)


def finish(run: Any) -> None:
    """Finish a W&B run, if there is one.

    Args:
        run (Any): The W&B run, or None.
    """
    if run is not None:
        run.finish()


def log_table(run: Any, key: str, rows: list[dict[str, Any]]) -> None:
    """Log a list of flat rows as one W&B table.

    Args:
        run (Any): The W&B run, or None.
        key (str): The key of the table.
        rows (list[dict[str, Any]]): The rows, all with the same keys.
    """
    if run is None or not rows:
        return

    columns: list[str | int] = list(rows[0])

    run.log({key: wandb.Table(columns=columns, data=[list(row.values()) for row in rows])})


def log_scores(
    run: Any, split: str, scores: dict[str, dict[str, Any]], predictions: dict[str, list[dict]]
) -> None:
    """Log the scores of one partition, per language, to a W&B run.

    Args:
        run (Any): The W&B run, or None.
        split (str): The partition name.
        scores (dict[str, dict[str, Any]]): The scores per language.
        predictions (dict[str, list[dict]]): The per-line predictions per language.
    """
    if run is None:
        return

    for lang, metrics in scores.items():
        prefix = f"{split}/{lang}"

        run.summary.update(
            {
                f"{prefix}/{key}": value
                for key, value in metrics.items()
                if isinstance(value, int | float)
            }
        )
        log_table(
            run,
            f"{prefix}/per_emotion",
            [{"emotion": emotion, **metrics["per_emotion"][emotion]} for emotion in EMOTIONS],
        )
        log_table(
            run,
            f"{prefix}/predictions",
            [
                {**row, "gold": ", ".join(row["gold"]), "predicted": ", ".join(row["predicted"])}
                for row in predictions[lang]
            ],
        )
