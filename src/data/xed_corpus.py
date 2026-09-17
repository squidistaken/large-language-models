from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.constants import SOURCE_LANG
from src.data.labels import XED_LABELS, LabelSpace
from src.utils.logger import Logger

PIVOT_PAIRS_FILE = f"pairs-{SOURCE_LANG}.txt"
PAIRS_FIELD_COUNT = 6
LABELLED_FIELD_COUNT = 2
MAX_REPORTED_LINES = 10


def make_align_id(source_doc: str, target_doc: str, source_id: str, target_id: str) -> str:
    """Build a XED alignment identifier.

    Args:
        source_doc (str): The English OPUS document.
        target_doc (str): The target-language OPUS document.
        source_id (str): The English sentence identifier.
        target_id (str): The target-language sentence identifier.

    Returns:
        str: An alignment identifier.
    """
    return f"{source_doc}|{target_doc}|{source_id}|{target_id}"


@dataclass(frozen=True)
class PairRow:
    """One subtitle alignment."""

    source_doc: str
    source_id: str
    source_text: str
    target_doc: str
    target_id: str
    target_text: str

    def align_id(self, source_doc: str | None = None, source_id: str | None = None) -> str:
        """Build the alignment identifier for this row.

        Args:
            source_doc (str | None): The English document to use instead of the row's source.
            source_id (str | None): The English sentence id to use instead of the row's source.

        Returns:
            str: An alignment identifier.
        """
        return make_align_id(
            source_doc if source_doc is not None else self.source_doc,
            self.target_doc,
            source_id if source_id is not None else self.source_id,
            self.target_id,
        )


@dataclass
class LabelledFile:
    """The usable rows of a XED `sentence\\tlabels` file, plus what was discarded."""

    rows: list[tuple[str, frozenset[int]]] = field(default_factory=list)
    malformed: list[int] = field(default_factory=list)
    unlabelled: int = 0


class XEDCorpus:
    """Resolves and reads the raw XED files for one source-target language pair."""

    def __init__(
        self,
        raw_dir: Path,
        target_lang: str,
        logger: Logger,
        labels: LabelSpace = XED_LABELS,
    ) -> None:
        """Initialise the class.

        Args:
            raw_dir (Path): The directory holding the downloaded XED files.
            target_lang (str): The assigned target language.
            logger (Logger): The logger.
            labels (LabelSpace): The label space to parse against.
        """
        self.raw_dir = raw_dir
        self.target_lang = target_lang
        self.logger = logger
        self.labels = labels

    @property
    def annotated_path(self) -> Path:
        """Get the path to the human-annotated English file.

        Returns:
            Path: A path.
        """
        return self.raw_dir / f"{SOURCE_LANG}-annotated.tsv"

    @property
    def projections_path(self) -> Path:
        """Get the path to the target language's projection file.

        Returns:
            Path: A path.
        """
        return self.raw_dir / f"{self.target_lang}-projections.tsv"

    @property
    def pairs_path(self) -> Path:
        """Get the path to the target language's subtitle-alignment pairs file.

        Returns:
            Path: A path.
        """
        return self.raw_dir / f"pairs-{self.target_lang}.txt"

    @property
    def pivot_path(self) -> Path:
        """Get the path to the Danish-to-English alignment file used to bridge pivot alignments.

        Returns:
            Path: A path.
        """
        return self.raw_dir / PIVOT_PAIRS_FILE

    def required_paths(self, use_pivot: bool) -> list[Path]:
        """List every raw file a preparation run reads.

        Args:
            use_pivot (bool): Whether pivot-bridged alignments are in use.

        Returns:
            list[Path]: A list of required paths.
        """
        required = [self.annotated_path, self.projections_path, self.pairs_path]

        if use_pivot:
            required.append(self.pivot_path)

        return required

    @staticmethod
    def iter_fields(path: Path) -> Iterator[tuple[int, list[str]]]:
        """Yield the tab-separated fields of each line, with its one-based line number.

        Args:
            path (Path): The file to read.

        Yields:
            tuple[int, list[str]]: A tuple containing line numbers and its fields.
        """
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                yield number, line.rstrip("\n").rstrip("\r").split("\t")

    def read_labelled(self, path: Path) -> LabelledFile:
        """Read a XED `sentence\\tlabels` file, dropping malformed and unlabelled rows.

        Args:
            path (Path): The file to read.

        Returns:
            LabelledFile: A collection of retained rows and the discard counts.
        """
        result = LabelledFile()

        for number, fields in self.iter_fields(path):
            if len(fields) != LABELLED_FIELD_COUNT:
                result.malformed.append(number)
                self.logger.debug(
                    f"{path.name}:{number} has {len(fields)} field(s), "
                    f"expected {LABELLED_FIELD_COUNT}; skipped"
                )

                continue

            text = fields[0].strip()
            label_ids = self.labels.parse_ids(fields[1])

            if not text or not label_ids:
                result.unlabelled += 1

                continue

            result.rows.append((text, label_ids))

        self._log_labelled(path, result)

        return result

    def _log_labelled(self, path: Path, result: LabelledFile) -> None:
        """Report what a labelled file yielded, summarising malformed lines rather than
        naming each one on the console.

        Args:
            path (Path): The file that was read.
            result (LabelledFile): The parsed result.
        """
        self.logger.info(
            f"Read {path.name}: {len(result.rows)} usable, {len(result.malformed)} malformed, "
            f"{result.unlabelled} unlabelled"
        )

        if not result.malformed:
            return

        shown = ", ".join(str(number) for number in result.malformed[:MAX_REPORTED_LINES])
        remaining = len(result.malformed) - MAX_REPORTED_LINES

        self.logger.info(
            f"{path.name} malformed line(s): {shown}"
            + (f", and {remaining} more" if remaining > 0 else "")
        )

    @staticmethod
    def iter_pairs(path: Path, target_lang: str) -> Iterator[PairRow]:
        """Yield alignment rows oriented so that `target` is the requested language.

        Args:
            path (Path): The pairs file to read.
            target_lang (str): The language to treat as the target side.

        Yields:
            PairRow: An iterator of one oriented alignment row.
        """
        for _, fields in XEDCorpus.iter_fields(path):
            if len(fields) != PAIRS_FIELD_COUNT:
                continue

            left_doc, right_doc, left_id, right_id, left_text, right_text = fields
            left = (left_doc, left_id, left_text.strip())
            right = (right_doc, right_id, right_text.strip())

            if right_doc.startswith(f"{target_lang}/"):
                yield PairRow(*left, *right)

            elif left_doc.startswith(f"{target_lang}/"):
                yield PairRow(*right, *left)
