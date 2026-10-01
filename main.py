import hydra
from omegaconf import DictConfig, OmegaConf

from src.cli.menu import run_cli
from src.utils.logger import Logger


@hydra.main(config_path="conf", config_name="main", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the program.

    Args:
        cfg (DictConfig): The dictionary object override.
    """
    logger = Logger(cfg.logging)
    logger.info("Run started")
    logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

    try:
        run_cli(cfg, logger)
    except Exception:
        logger.exception("Run failed with an unhandled exception")
        raise

    logger.info("Run finished")


if __name__ == "__main__":
    main()
