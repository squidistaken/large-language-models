from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.constants import (
    DEFAULT_SEED,
    DEFAULT_SPLITS,
    DEFAULT_TARGET_LANG,
    EMOTIONS,
    LANGUAGE_TO_BAND,
    PIVOT_LANG,
    PROCESSED_DIR,
    RAW_DIR,
    SOURCE_LANG,
)
from src.data.alignment import AlignmentBuilder, AlignmentStats
from src.data.formatting import USER_TEMPLATE
from src.data.splitting import STRATIFIER, SplitAssigner, SplitStats
from src.data.writing import SplitWriter
from src.data.xed_corpus import XEDCorpus
from src.utils.files import checksum, read_json
from src.utils.logger import Logger

CHECKSUM_PREFIX = 12


def pair_name(target_lang: str) -> str:
    """Name the directory holding one source-target pair's splits.

    Args:
        target_lang (str): The target language.

    Returns:
        str: A pair name.
    """
    return f"{SOURCE_LANG}-{target_lang}"


@dataclass
class PreparationSummary:
    """Counts describing a preparation run, rendered by the CLI and stored in the manifest."""

    target_lang: str
    seed: int
    splits: tuple[float, ...]
    skipped: bool = False
    alignment: AlignmentStats = field(default_factory=AlignmentStats)
    split: SplitStats = field(default_factory=SplitStats)
    label_support: dict[str, dict[str, int]] = field(default_factory=dict)
    raw_checksums: dict[str, str] = field(default_factory=dict)
    mismatches: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest: dict) -> "PreparationSummary":
        """Rebuild the summary of an earlier run from its manifest.

        Args:
            manifest (dict): The manifest the earlier run wrote.

        Returns:
            PreparationSummary: A summary of the run.
        """
        counts = manifest.get("counts", {})

        alignment = AlignmentStats(
            target_rows=counts.get("target_rows", 0),
            malformed_rows=counts.get("malformed_rows", 0),
            unlabelled_rows=counts.get("unlabelled_rows", 0),
            direct_alignments=counts.get("direct_alignments", 0),
            bridged_alignments=counts.get("bridged_alignments", 0),
            resolved_units=counts.get("resolved_units", 0),
            unresolved_rows=counts.get("unresolved_rows", 0),
            duplicate_rows=counts.get("duplicate_rows", 0),
            gold_source_labels=counts.get("gold_source_labels", 0),
            pivot_anchored_ids=counts.get("pivot_anchored_ids", 0),
            units_by_hops=manifest.get("units_by_alignment_hops", {}),
        )
        split = SplitStats(
            leakage_groups=counts.get("multi_unit_groups", 0),
            split_sizes=manifest.get("split_sizes", {}),
        )

        return cls(
            target_lang=manifest.get("target_lang", ""),
            seed=manifest.get("seed", DEFAULT_SEED),
            splits=tuple(manifest.get("splits", DEFAULT_SPLITS)),
            alignment=alignment,
            split=split,
            label_support=manifest.get("label_support", {}),
            raw_checksums=manifest.get("raw_checksums", {}),
        )

    @property
    def duplicate_rows(self) -> int:
        """Duplicate pairs dropped, summed across both deduplication passes.

        Returns:
            int: A count.
        """
        return self.alignment.duplicate_rows + self.split.duplicate_rows


class XEDPreparer:
    """Class that builds a leakage-controlled stratified split of matched XED language pairs."""

    def __init__(self, cfg: DictConfig, logger: Logger) -> None:
        """Initialise the class.

        Args:
            cfg (DictConfig): The dictionary object override.
            logger (Logger): The logger.
        """
        self.target_lang = cfg.get("target_lang", DEFAULT_TARGET_LANG)
        self.raw_dir = Path(cfg.get("raw_dir", RAW_DIR))
        self.out_dir = Path(cfg.get("out_dir", PROCESSED_DIR)) / pair_name(self.target_lang)
        self.splits = tuple(float(value) for value in cfg.get("splits", DEFAULT_SPLITS))
        self.seed = int(cfg.get("seed", DEFAULT_SEED))
        self.use_pivot = cfg.get("use_pivot", True)
        self.overwrite = cfg.get("overwrite", False)
        self.logger = logger
        self.corpus = XEDCorpus(self.raw_dir, self.target_lang, logger)
        self.builder = AlignmentBuilder(self.corpus, logger, use_pivot=self.use_pivot)
        self.assigner = SplitAssigner(logger, splits=self.splits, seed=self.seed)
        self.writer = SplitWriter(self.out_dir, self.target_lang, logger)

        self.summary = PreparationSummary(
            target_lang=self.target_lang,
            seed=self.seed,
            splits=self.splits,
            alignment=self.builder.stats,
            split=self.assigner.stats,
        )

    def prepare(self) -> PreparationSummary:
        """Build and write the train/development/test split.

        Returns:
            PreparationSummary: A collection of counts describing the run.

        Raises:
            FileNotFoundError: If a required raw input file is missing.
            ValueError: If no target line could be aligned to an English counterpart.
        """
        self._check_inputs()

        existing = self.writer.existing()

        if existing and not self.overwrite:
            self.logger.info(
                f"Skipped preparation: {len(existing)} split file(s) already exist under "
                f"{self.out_dir}"
            )

            return self._load_existing()

        units = self.assigner.deduplicate(self.builder.build())

        if not units:
            raise ValueError(self.builder.no_units_message())

        assignments = self.assigner.assign(units)

        self.summary.label_support = self.writer.write(units, assignments)
        self.writer.write_manifest(self._manifest())

        return self.summary

    def _load_existing(self) -> PreparationSummary:
        """Summarise the splits already on disk, noting where they differ from this request.

        Returns:
            PreparationSummary: An earlier run's summary, marked as skipped.
        """
        if not self.writer.manifest_path.exists():
            self.summary.skipped = True
            self.summary.mismatches = {"manifest": ("missing", self.writer.manifest_path.name)}

            return self.summary

        manifest = read_json(self.writer.manifest_path)
        summary = PreparationSummary.from_manifest(manifest)
        requested = {
            "target_lang": self.target_lang,
            "seed": self.seed,
            "splits": list(self.splits),
            "use_pivot": self.use_pivot,
            "prompt_sha256": self.writer.formatter.prompt_sha256,
        }

        summary.skipped = True
        summary.mismatches = {
            key: (manifest.get(key), value)
            for key, value in requested.items()
            if manifest.get(key) != value
        }

        if "prompt_sha256" in summary.mismatches:
            previous, current = summary.mismatches["prompt_sha256"]
            summary.mismatches["prompt_sha256"] = (
                str(previous)[:CHECKSUM_PREFIX] if previous else None,
                current[:CHECKSUM_PREFIX],
            )

        same_target = manifest.get("target_lang") == self.target_lang

        for name, digest in self.summary.raw_checksums.items():
            previous = summary.raw_checksums.get(name)

            if previous is None and not same_target:
                continue

            if previous != digest:
                summary.mismatches[name] = (
                    f"sha256 {previous[:CHECKSUM_PREFIX]}" if previous else "not used",
                    f"sha256 {digest[:CHECKSUM_PREFIX]}",
                )

        if summary.mismatches:
            self.logger.info(f"Existing splits differ from this request: {summary.mismatches}")

        return summary

    def _check_inputs(self) -> None:
        """Validate that every required raw input is present and checksum it.

        Raises:
            FileNotFoundError: If a required raw input file is missing.
        """
        required = self.corpus.required_paths(self.use_pivot)
        missing = [str(path) for path in required if not path.exists()]

        if missing:
            raise FileNotFoundError(
                f"Missing raw input(s): {', '.join(missing)}. Run the download action first."
            )

        for path in self.logger.track(
            required, desc="Checking inputs", desc_fn=lambda path: f"Checking {path.name}"
        ):
            self.summary.raw_checksums[path.name] = checksum(path)

    def _manifest(self) -> dict[str, Any]:
        """Build the manifest recording every choice needed to reproduce the run.

        Returns:
            dict: A dictionary containing the manifest document.
        """
        alignment, split = self.summary.alignment, self.summary.split

        return {
            "target_lang": self.target_lang,
            "target_band": LANGUAGE_TO_BAND.get(self.target_lang),
            "source_lang": SOURCE_LANG,
            "seed": self.seed,
            "splits": list(self.splits),
            "label_order": list(EMOTIONS),
            "use_pivot": self.use_pivot,
            "pivot_lang": PIVOT_LANG if self.use_pivot else None,
            "stratifier": STRATIFIER,
            "prompt_sha256": self.writer.formatter.prompt_sha256,
            "prompt": {
                "system": self.writer.formatter.system_prompt,
                "user_template": USER_TEMPLATE,
            },
            "counts": {
                "target_rows": alignment.target_rows,
                "malformed_rows": alignment.malformed_rows,
                "unlabelled_rows": alignment.unlabelled_rows,
                "resolved_units": alignment.resolved_units,
                "unresolved_rows": alignment.unresolved_rows,
                "duplicate_rows": self.summary.duplicate_rows,
                "multi_unit_groups": split.leakage_groups,
                "gold_source_labels": alignment.gold_source_labels,
                "direct_alignments": alignment.direct_alignments,
                "bridged_alignments": alignment.bridged_alignments,
                "pivot_anchored_ids": alignment.pivot_anchored_ids,
            },
            "units_by_alignment_hops": alignment.units_by_hops,
            "split_sizes": split.split_sizes,
            "label_support": self.summary.label_support,
            "raw_checksums": self.summary.raw_checksums,
        }
