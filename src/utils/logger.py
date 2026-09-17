import logging
import warnings
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from omegaconf import DictConfig
from rich.logging import RichHandler
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

# tqdm.rich is experimental.
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

DEFAULT_LOG_DIR = "logs"

T = TypeVar("T")


class Logger:
    """Specialised logging class."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The dictionary object override.
        """
        self.prefix = cfg.get("prefix", default_value="")
        file_level = cfg.get("level", "INFO")
        verbose = cfg.get("verbose", False)
        log_dir = Path(cfg.get("log_dir", DEFAULT_LOG_DIR))

        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / f"{datetime.now().astimezone():%Y%m%d_%H%M%S}.log"
        console_handler = RichHandler(
            show_time=False, show_level=False, show_path=False, rich_tracebacks=True
        )

        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.setLevel(logging.INFO if verbose else logging.WARNING)

        file_handler = logging.FileHandler(self.log_file)

        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        file_handler.setLevel(file_level)
        logging.basicConfig(
            level=logging.DEBUG, handlers=[console_handler, file_handler], force=True
        )

        self._logger = logging.getLogger("xed")

    def _fmt(self, msg: str) -> str:
        """Prepend a configured prefix to a message.

        Args:
            msg (str): The raw log message.

        Returns:
            str: A message with a prefix applied.
        """
        return f"{self.prefix} {msg}" if self.prefix else msg

    def info(self, msg: str) -> None:
        """Log an info-level message.

        Args:
            msg(str): The raw log message.
        """
        self._logger.info(self._fmt(msg))

    def warning(self, msg: str) -> None:
        """Log a warning-level message.

        Args:
            msg(str): The raw log message.
        """
        self._logger.warning(self._fmt(msg))

    def error(self, msg: str) -> None:
        """Log an error-level message.

        Args:
            msg(str): The raw log message.
        """
        self._logger.error(self._fmt(msg))

    def exception(self, msg: str) -> None:
        """Log an error-level message including the current exception's
        traceback.

        This should be called from inside an `except` block.

        Args:
            msg(str): The raw log message.
        """
        self._logger.exception(self._fmt(msg))

    def debug(self, msg: str) -> None:
        """Log a debug-level message.

        Args:
            msg(str): The raw log message.
        """
        self._logger.debug(self._fmt(msg))

    def track(
        self,
        iterable: Iterable[T],
        desc: str = "",
        desc_fn: Callable[[T], str] | None = None,
    ) -> Iterator[T]:
        """Wrap an iterable with a Rich-rendered progress bar.

        Args:
            iterable (Iterable[T]): The iterable to progress-track.
            desc (str): The initial label shown next to the bar. Defaults to "".
            desc_fn (Callable[[T], str] | None): The callback to update the bar's description.
                                                 Defaults to None.

        Yields:
            T: An iteration of the progress bar.
        """
        bar = tqdm(iterable, desc=desc)

        for item in bar:
            if desc_fn is not None:
                bar.set_description(desc_fn(item))

            yield item
