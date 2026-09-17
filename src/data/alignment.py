import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.constants import GOLD, PIVOT_LANG, PROJECTED, SOURCE_LANG
from src.data.labels import XED_LABELS, LabelSpace
from src.data.xed_corpus import XEDCorpus
from src.utils.logger import Logger
from src.utils.text import normalise

# XED's English annotations are NER-anonymised.
PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")
DIRECT_HOPS = 1
BRIDGED_HOPS = 2


@dataclass(frozen=True)
class AlignmentUnit:
    """One aligned English-target subtitle pair, keyed by its XED alignment identifier."""

    align_id: str
    source_text: str
    source_labels: frozenset[int]
    source_label_source: str
    target_text: str
    target_labels: frozenset[int]
    alignment_hops: int = DIRECT_HOPS
    n_alignments: int = 1


@dataclass(frozen=True)
class Candidate:
    """One English line that could stand as the counterpart of a target line."""

    hops: int
    align_id: str
    text: str

    def order(self) -> tuple[int, str]:
        """Sort key placing the shortest, then lowest-identified, alignment first.

        Returns:
            tuple[int, str]: A tuple containing the sort key.
        """
        return self.hops, self.align_id


@dataclass
class AlignmentStats:
    """Counts describing how target lines were resolved to English counterparts."""

    target_rows: int = 0
    malformed_rows: int = 0
    unlabelled_rows: int = 0
    direct_alignments: int = 0
    bridged_alignments: int = 0
    resolved_units: int = 0
    unresolved_rows: int = 0
    duplicate_rows: int = 0
    gold_source_labels: int = 0
    units_by_hops: dict[str, int] = field(default_factory=dict)


class GoldEnglishIndex:
    """Lookup over XED's human-annotated English lines, tolerant of NER anonymisation."""

    def __init__(self, logger: Logger) -> None:
        """Initialise the class.

        Args:
            logger (Logger): The logger.
        """
        self.exact: dict[str, frozenset[int]] = {}
        self.buckets: dict[str, list[tuple[re.Pattern, frozenset[int]]]] = defaultdict(list)
        self.logger = logger

    def add(self, text: str, label_ids: frozenset[int]) -> None:
        """Index one annotated English line.

        Args:
            text (str): The annotated line, possibly containing NER placeholders.
            label_ids (frozenset[int]): The gold label ids.
        """
        if not PLACEHOLDER_PATTERN.search(text):
            self.exact.setdefault(normalise(text), label_ids)

            return

        segments = [normalise(segment) for segment in PLACEHOLDER_PATTERN.split(text)]

        # A line that is nothing but a placeholder would match anything, so it is unusable.
        if not any(segments):
            return

        pattern = re.compile(".+?".join(re.escape(segment) for segment in segments))

        self.buckets[segments[0].split(" ")[0]].append((pattern, label_ids))

    def lookup(self, key: str) -> frozenset[int] | None:
        """Look up a normalised English line.

        Args:
            key (str): The normalised English line.

        Returns:
            frozenset[int] | None: A set of gold label ids, or None if the line is not annotated.
        """
        if key in self.exact:
            return self.exact[key]

        # The empty bucket holds lines that begin with a placeholder, whose leading word is
        # unknown, so it is always a candidate.
        for pattern, label_ids in self.buckets.get(key.split(" ")[0], []) + self.buckets.get(
            "", []
        ):
            if pattern.fullmatch(key):
                return label_ids

        return None

    def log_summary(self) -> None:
        """Report how many lines were indexed each way."""
        patterns = sum(len(bucket) for bucket in self.buckets.values())

        self.logger.info(
            f"Gold English index: {len(self.exact)} exact, {patterns} anonymised patterns"
        )


class AlignmentBuilder:
    """Class that resolves each target-language line to the English line XED projected it from."""

    def __init__(
        self,
        corpus: XEDCorpus,
        logger: Logger,
        use_pivot: bool = True,
        labels: LabelSpace = XED_LABELS,
    ) -> None:
        """Initialise the class.

        Args:
            corpus (XEDCorpus): The raw XED files to read.
            logger (Logger): The logger.
            use_pivot (bool): Whether to bridge Danish-aligned rows back to English.
            labels (LabelSpace): The label space in use.
        """
        self.corpus = corpus
        self.logger = logger
        self.use_pivot = use_pivot
        self.labels = labels
        self.stats = AlignmentStats()

    def build(self) -> list[AlignmentUnit]:
        """Resolve every target-language line that has an English counterpart.

        Returns:
            list[AlignmentUnit]: A list of resolved alignment units.
        """
        index = self._build_gold_index()
        candidates = self._collect_candidates()
        projections = self.corpus.read_labelled(self.corpus.projections_path)

        self.stats.target_rows = len(projections.rows)
        self.stats.malformed_rows = len(projections.malformed)
        self.stats.unlabelled_rows = projections.unlabelled

        units: list[AlignmentUnit] = []
        seen: set[str] = set()

        for text, label_ids in self.logger.track(projections.rows, desc="Aligning"):
            matches = candidates.get(normalise(text))

            if not matches:
                self.stats.unresolved_rows += 1

                continue

            unit = self._resolve(text, label_ids, matches, index)

            if unit.align_id in seen:
                self.stats.duplicate_rows += 1

                continue

            seen.add(unit.align_id)
            units.append(unit)

        self._record_stats(units)

        return units

    def _resolve(
        self,
        text: str,
        label_ids: frozenset[int],
        matches: list[Candidate],
        index: GoldEnglishIndex,
    ) -> AlignmentUnit:
        """Choose one English counterpart for a target line and build its unit.

        Args:
            text (str): The target-language line.
            label_ids (frozenset[int]): Its projected label ids.
            matches (list[Candidate]): The available English counterparts.
            index (GoldEnglishIndex): The gold English lookup.

        Returns:
            AlignmentUnit: A resolved unit.
        """
        ordered = sorted(matches, key=Candidate.order)
        chosen = ordered[0]
        source_labels, source = label_ids, PROJECTED

        for candidate in ordered:
            gold = index.lookup(normalise(candidate.text))

            if gold is not None:
                chosen = candidate
                source_labels, source = gold, GOLD

                break

        return AlignmentUnit(
            align_id=chosen.align_id,
            source_text=chosen.text,
            source_labels=source_labels,
            source_label_source=source,
            target_text=text,
            target_labels=label_ids,
            alignment_hops=chosen.hops,
            n_alignments=len(matches),
        )

    def _build_gold_index(self) -> GoldEnglishIndex:
        """Index XED's human-annotated English file.

        Returns:
            GoldEnglishIndex: A populated index.
        """
        index = GoldEnglishIndex(self.logger)
        annotated = self.corpus.read_labelled(self.corpus.annotated_path)

        for text, label_ids in annotated.rows:
            index.add(text, label_ids)

        index.log_summary()

        return index

    def _read_pivot(self) -> dict[str, tuple[str, str, str]]:
        """Index the Danish-to-English alignments that bridge pivot rows back to English.

        Returns:
            dict[str, tuple[str, str, str]]: A dictionary containing an English document, identifier
                                             and text per normalised pivot sentence.
        """
        pivot: dict[str, tuple[str, str, str]] = {}

        # Oriented with English as the target, so the source side is the pivot language.
        for row in self.corpus.iter_pairs(self.corpus.pivot_path, SOURCE_LANG):
            if row.source_doc.startswith(f"{PIVOT_LANG}/"):
                pivot.setdefault(
                    normalise(row.source_text),
                    (row.target_doc, row.target_id, row.target_text),
                )

        self.logger.info(
            f"Read {self.corpus.pivot_path.name}: {len(pivot)} pivot-to-English bridges"
        )

        return pivot

    def _collect_candidates(self) -> dict[str, list[Candidate]]:
        """Collect every English line that aligns to each target-language line.

        Returns:
            dict[str, list[Candidate]]: A dictionary containing the alignment candidates per
                                        normalised target sentence.
        """
        candidates: dict[str, list[Candidate]] = defaultdict(list)
        pivot = self._read_pivot() if self.use_pivot else {}

        for row in self.corpus.iter_pairs(self.corpus.pairs_path, self.corpus.target_lang):
            key = normalise(row.target_text)

            if row.source_doc.startswith(f"{SOURCE_LANG}/"):
                candidates[key].append(Candidate(DIRECT_HOPS, row.align_id(), row.source_text))
                self.stats.direct_alignments += 1

                continue

            bridge = pivot.get(normalise(row.source_text))

            if bridge is None:
                continue

            en_doc, en_id, en_text = bridge

            candidates[key].append(Candidate(BRIDGED_HOPS, row.align_id(en_doc, en_id), en_text))
            self.stats.bridged_alignments += 1

        self.logger.info(
            f"Read {self.corpus.pairs_path.name}: {self.stats.direct_alignments} direct and "
            f"{self.stats.bridged_alignments} pivot-bridged English alignments over "
            f"{len(candidates)} distinct target lines"
        )

        return candidates

    def _record_stats(self, units: list[AlignmentUnit]) -> None:
        """Fold the resolved units into the run statistics.

        Args:
            units (list[AlignmentUnit]): A list of resolved units.
        """
        self.stats.resolved_units = len(units)
        self.stats.gold_source_labels = sum(1 for unit in units if unit.source_label_source == GOLD)
        self.stats.units_by_hops = {
            str(hops): count
            for hops, count in sorted(Counter(unit.alignment_hops for unit in units).items())
        }

        self.logger.info(
            f"Resolved {len(units)}/{self.stats.target_rows} target lines to alignment "
            f"identifiers ({self.stats.unresolved_rows} unresolved, "
            f"{self.stats.duplicate_rows} duplicate); {self.stats.gold_source_labels} have gold "
            "English annotations"
        )

    def no_units_message(self) -> str:
        """Explain why no alignment unit survived, with the most likely remedy.

        Returns:
            str: A message to raise.
        """
        message = (
            f"No {self.corpus.target_lang} line could be aligned to an English counterpart "
            f"({self.stats.direct_alignments} direct and {self.stats.bridged_alignments} "
            "bridged alignments were available)."
        )

        if not self.use_pivot:
            return (
                f"{message} Many XED annotations are recorded against {PIVOT_LANG} subtitle "
                "ids, so some languages have few or no direct English alignments. Re-run with "
                "prepare.use_pivot=true."
            )

        return (
            f"{message} Check that {self.corpus.pairs_path.name} and "
            f"{self.corpus.pivot_path.name} are the files XED distributes and were not "
            "truncated on download."
        )
