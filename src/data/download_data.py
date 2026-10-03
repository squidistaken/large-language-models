import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import certifi
from omegaconf import DictConfig

from src.constants import ALL_LANGUAGES, ANNOTATED_LANGS, SOURCE_LANG
from src.utils.logger import Logger

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master"
REQUEST_TIMEOUT_SECONDS = 30
# Trust certifi's certificates.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
DEFAULT_ANNOTATED_DIR = "AnnotatedData"
DEFAULT_PROJECTIONS_DIR = "Projections"
DEFAULT_PAIRS_DIR = "subtitle-retrieval/students"
ANNOTATIONS_FILE = "student_annotations.csv"

# Weird mistake from the repository.
PROJECTION_FILENAME_OVERRIDES = {"tr": "tr-projections2.tsv"}
PROJECTION_REMOTE_CODE_OVERRIDES = {"pt_br": "br"}


@dataclass
class DownloadSummary:
    """The outcome of a download run."""

    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def nothing_to_do(self) -> bool:
        """Get whether every requested file was already present.

        Returns:
            bool: `True` when nothing was downloaded and nothing failed.
        """
        return bool(self.skipped) and not self.downloaded and not self.failed


class XEDDownloader:
    """Class to download XED annotated and projection files for configured languages."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The dictionary object override.
            logger (Logger): The logger.
        """
        languages = cfg.languages

        if isinstance(languages, str):
            languages = [languages]

        languages = list(languages)

        if any(lang.lower() == "all" for lang in languages):
            languages = ALL_LANGUAGES

        if SOURCE_LANG not in languages:
            languages = [SOURCE_LANG, *languages]

        self.out_dir = Path(cfg.out_dir)
        self.languages = languages
        self.force = cfg.get("force", False)
        self.base_url = cfg.get("base_url", DEFAULT_BASE_URL)
        self.annotated_dir = cfg.get("annotated_dir", DEFAULT_ANNOTATED_DIR)
        self.projections_dir = cfg.get("projections_dir", DEFAULT_PROJECTIONS_DIR)
        self.pairs_dir = cfg.get("pairs_dir", DEFAULT_PAIRS_DIR)
        self.logger = logger

    def fetch_files(self) -> dict[str, str]:
        """Fetch files to download.

        Returns:
            dict[str, str]: A dictionary containing the files to download.
        """
        files = {}

        for lang in self.languages:
            if lang in ANNOTATED_LANGS:
                files[f"{lang}-annotated.tsv"] = (
                    f"{self.base_url}/{self.annotated_dir}/{lang}-annotated.tsv"
                )
                files[f"neu_{lang}.txt"] = f"{self.base_url}/{self.annotated_dir}/neu_{lang}.txt"

            if lang != SOURCE_LANG:
                remote_code = PROJECTION_REMOTE_CODE_OVERRIDES.get(lang, lang)
                remote_filename = PROJECTION_FILENAME_OVERRIDES.get(
                    remote_code, f"{remote_code}-projections.tsv"
                )
                local_filename = f"{lang}-projections.tsv"
                files[local_filename] = f"{self.base_url}/{self.projections_dir}/{remote_filename}"

            if lang in ALL_LANGUAGES:
                files[f"pairs-{lang}.txt"] = f"{self.base_url}/{self.pairs_dir}/pairs-{lang}.txt"

            if lang == SOURCE_LANG:
                files[ANNOTATIONS_FILE] = f"{self.base_url}/{self.pairs_dir}/{ANNOTATIONS_FILE}"

        return files

    def download(self) -> DownloadSummary:
        """Download files.

        Skips files that already exist locally unless `self.force == True`.

        Returns:
            DownloadSummary: A collection of files downloaded, skipped, and failed.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        summary = DownloadSummary()
        files = self.fetch_files().items()

        for filename, url in self.logger.track(
            files, desc="Downloading", desc_fn=lambda item: f"Downloading {item[0]}"
        ):
            dest = self.out_dir / filename

            if dest.exists() and not self.force:
                summary.skipped.append(filename)
                self.logger.debug(f"Skipped {filename} (already exists)")

                continue

            try:
                self._fetch(url, dest)
                summary.downloaded.append(filename)
                self.logger.info(f"Downloaded {filename}")

            except RuntimeError as e:
                summary.failed.append(filename)
                self.logger.error(str(e))

        self.logger.info(
            f"Downloaded {len(summary.downloaded)}, skipped {len(summary.skipped)}, "
            f"failed {len(summary.failed)}"
        )

        return summary

    @staticmethod
    def _fetch(url: str, dest: Path) -> None:
        """Download a single file to disk.

        Args:
            url (str): The URL to download the file from.
            dest (Path): The local path to write the downloaded content to.

        Raises:
            RuntimeError: If the request fails.
        """
        req = urllib.request.Request(url, headers={"User-Agent": "xed-downloader"})

        try:
            with (
                urllib.request.urlopen(
                    req, timeout=REQUEST_TIMEOUT_SECONDS, context=SSL_CONTEXT
                ) as resp,
                open(dest, "wb") as out_f,
            ):
                out_f.write(resp.read())

        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Failed to fetch {url}: HTTP {e.code}") from e

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch {url}: {e.reason}") from e
