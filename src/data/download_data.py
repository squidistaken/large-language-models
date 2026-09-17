import urllib.error
import urllib.request
from pathlib import Path

from omegaconf import DictConfig

from src.utils.logger import Logger

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_ANNOTATED_DIR = "AnnotatedData"
DEFAULT_PROJECTIONS_DIR = "Projections"
ANNOTATED_LANGS = {"en", "fi"}
PROJECTION_ONLY_LANGS = {
    "ar",
    "bg",
    "bs",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "eu",
    "fa",
    "fr",
    "gl",
    "he",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "mk",
    "ml",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    "th",
    "pt_br",
    "tr",
    "vi",
    "zh_cn",
    "zh_tw",
    "zh_yue",
}
ALL_LANGUAGES = sorted(ANNOTATED_LANGS | PROJECTION_ONLY_LANGS)

# Weird mistake from the repository.
PROJECTION_FILENAME_OVERRIDES = {"tr": "tr-projections2.tsv"}
PROJECTION_REMOTE_CODE_OVERRIDES = {
    "pt_br": "br",
    "zh_cn": "cn",
    "zh_tw": "tw",
    "zh_yue": "zh",
}
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bs": "Bosnian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "he": "Hebrew",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt_br": "Portuguese (Brazil)",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh_cn": "Chinese (Simplified)",
    "zh_tw": "Chinese (Traditional)",
    "zh_yue": "Chinese (Cantonese)",
}


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

        self.out_dir = Path(cfg.out_dir)
        self.languages = languages
        self.force = cfg.get("force", False)
        self.base_url = cfg.get("base_url", DEFAULT_BASE_URL)
        self.annotated_dir = cfg.get("annotated_dir", DEFAULT_ANNOTATED_DIR)
        self.projections_dir = cfg.get("projections_dir", DEFAULT_PROJECTIONS_DIR)
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

            else:
                remote_code = PROJECTION_REMOTE_CODE_OVERRIDES.get(lang, lang)
                remote_filename = PROJECTION_FILENAME_OVERRIDES.get(
                    remote_code, f"{remote_code}-projections.tsv"
                )
                local_filename = f"{lang}-projections.tsv"
                files[local_filename] = f"{self.base_url}/{self.projections_dir}/{remote_filename}"

        return files

    def download(self) -> list[str]:
        """Download files.

        Skips files that already exist locally unless `self.force == True`.

        Returns:
            list[str]: A list of failed downloaded files.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)

        failed = []
        files = self.fetch_files().items()

        for filename, url in self.logger.track(
            files, desc="Downloading", desc_fn=lambda item: f"Downloading {item[0]}"
        ):
            dest = self.out_dir / filename

            if dest.exists() and not self.force:
                self.logger.debug(f"Skipped {filename} (already exists)")

                continue

            try:
                self._fetch(url, dest)
                self.logger.info(f"Downloaded {filename}")

            except RuntimeError as e:
                self.logger.error(str(e))
                failed.append(filename)

        return failed

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
                urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp,
                open(dest, "wb") as out_f,
            ):
                out_f.write(resp.read())

        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Failed to fetch {url}: HTTP {e.code}") from e

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch {url}: {e.reason}") from e
