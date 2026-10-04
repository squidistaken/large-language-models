from collections import Counter
from pathlib import Path
from typing import Any

from src.constants import PROJECTED, SOURCE_LANG, SPLIT_NAMES
from src.data.alignment import AlignmentUnit
from src.data.formatting import ConversationFormatter
from src.data.labels import XED_LABELS, LabelSpace
from src.utils.files import write_json, write_jsonl
from src.utils.logger import Logger

MANIFEST_FILE = "split_manifest.json"


def pair_name(target_lang: str) -> str:
    """Get the name of the directory holding one source-target pair's splits.

    Args:
        target_lang (str): The target language.

    Returns:
        str: A pair name.
    """
    return f"{SOURCE_LANG}-{target_lang}"


class SplitWriter:
    """Class that writes the canonical and conversational split files, and the reproducibility
    manifest."""

    def __init__(
        self,
        out_dir: Path,
        target_lang: str,
        logger: Logger,
        labels: LabelSpace = XED_LABELS,
        formatter: ConversationFormatter | None = None,
    ) -> None:
        """Initialise the class.

        Args:
            out_dir (Path): The directory to write the splits into.
            target_lang (str): The assigned target language.
            logger (Logger): The logger.
            labels (LabelSpace): The label space in use.
            formatter (ConversationFormatter | None): The conversational renderer.
        """
        self.out_dir = out_dir
        self.target_lang = target_lang
        self.logger = logger
        self.labels = labels
        self.formatter = formatter or ConversationFormatter(labels)

    @classmethod
    def for_pair(cls, data_dir: str | Path, target_lang: str, logger: Logger) -> "SplitWriter":
        """Get the writer of one source-target pair's splits.

        Args:
            data_dir (str | Path): The folder holding every pair's splits.
            target_lang (str): The target language.
            logger (Logger): The logger.

        Returns:
            SplitWriter: A split writer.
        """
        return cls(Path(data_dir) / pair_name(target_lang), target_lang, logger)

    @property
    def languages(self) -> tuple[str, str]:
        """Get the two languages written, source first.

        Returns:
            tuple[str, str]: A tuple containing language codes.
        """
        return SOURCE_LANG, self.target_lang

    @property
    def manifest_path(self) -> Path:
        """Get the path to the reproducibility manifest.

        Returns:
            Path: A path.
        """
        return self.out_dir / MANIFEST_FILE

    def existing(self) -> list[Path]:
        """List the canonical split files that are already present.

        Returns:
            list[Path]: A list of existing split files.
        """
        return [
            path
            for lang in self.languages
            for split in SPLIT_NAMES
            for path in (self.canonical_path(lang, split),)
            if path.exists()
        ]

    def canonical_path(self, lang: str, split: str) -> Path:
        """Get the path to one canonical split file.

        Args:
            lang (str): The language.
            split (str): The partition name.

        Returns:
            Path: A path.
        """
        return self.out_dir / lang / f"{split}.jsonl"

    def chat_path(self, lang: str, split: str) -> Path:
        """Get the path to one conversational split file.

        Args:
            lang (str): The language.
            split (str): The partition name.

        Returns:
            Path: A path.
        """
        return self.out_dir / lang / f"{split}.chat.jsonl"

    def write(
        self, units: list[AlignmentUnit], assignments: list[str]
    ) -> dict[str, dict[str, int]]:
        """Write every split file for both languages.

        Args:
            units (list[AlignmentUnit]): The deduplicated units.
            assignments (list[str]): The partition name for each unit.

        Returns:
            dict[str, dict[str, int]]: A dictionary containing per-label support per partition, for
                                       the target side.
        """
        by_split: dict[str, list[AlignmentUnit]] = {name: [] for name in SPLIT_NAMES}

        for unit, split in zip(units, assignments, strict=True):
            by_split[split].append(unit)

        targets = [(lang, split) for lang in self.languages for split in SPLIT_NAMES]

        for lang, split in self.logger.track(
            targets, desc="Writing", desc_fn=lambda item: f"Writing {item[0]}/{item[1]}"
        ):
            rows = [self.record(unit, lang) for unit in by_split[split]]

            write_jsonl(self.canonical_path(lang, split), rows)
            write_jsonl(
                self.chat_path(lang, split), [self.formatter.conversation(row) for row in rows]
            )
            self.logger.info(f"Wrote {len(rows)} rows to {self.canonical_path(lang, split)}")

        return {split: self._support(by_split[split]) for split in SPLIT_NAMES}

    def record(self, unit: AlignmentUnit, lang: str) -> dict[str, Any]:
        """Build one canonical record for a unit in a given language.

        Args:
            unit (AlignmentUnit): The alignment unit.
            lang (str): The language to emit.

        Returns:
            dict[str, Any]: A canonical record.
        """
        is_source = lang == SOURCE_LANG
        text = unit.source_text if is_source else unit.target_text
        label_ids = unit.source_labels if is_source else unit.target_labels
        label_source = unit.source_label_source if is_source else PROJECTED

        return {
            "align_id": unit.align_id,
            "lang": lang,
            "text": text,
            "label_ids": sorted(label_ids),
            "labels": self.labels.to_names(label_ids),
            "vector": self.labels.to_vector(label_ids),
            "label_source": label_source,
            "alignment_hops": unit.alignment_hops,
            "n_alignments": unit.n_alignments,
        }

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write the manifest recording every choice needed to reproduce the run.

        Args:
            manifest (dict[str, Any]): The manifest document.
        """
        write_json(self.manifest_path, manifest)
        self.logger.info(f"Wrote manifest to {self.manifest_path}")

    def _support(self, units: list[AlignmentUnit]) -> dict[str, int]:
        """Count per-label support over the target side of a partition.

        Args:
            units (list[AlignmentUnit]): The units in the partition.

        Returns:
            dict[str, int]: A dictionary containing the support per label.
        """
        support = Counter()

        for unit in units:
            support.update(self.labels.to_names(unit.target_labels))

        return {name: support[name] for name in self.labels.names}
