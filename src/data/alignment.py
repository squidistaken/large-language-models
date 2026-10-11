import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.constants import GOLD, PIVOT_LANG, PROJECTED, SOURCE_LANG
from src.data.labels import XED_LABELS, LabelSpace
from src.data.xed_corpus import AnnotationRow, PairRow, XEDCorpus
from src.utils.logger import Logger
from src.utils.text import normalise

# XED's English annotations are NER-anonymised.
PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_]+\]")
DIGIT_PATTERN = re.compile(r"\d")
DIRECT_HOPS = 1
BRIDGED_HOPS = 2


@dataclass(frozen=True)
class AlignmentUnit:
    """The English and target-language lines that XED aligned, with their labels and the alignment's
    provenance."""

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
    """The English side of a target-language line's alignment, with its distance from the labels."""

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
    """The statistics of a run of the alignment builder."""

    target_rows: int = 0
    malformed_rows: int = 0
    unlabelled_rows: int = 0
    direct_alignments: int = 0
    bridged_alignments: int = 0
    resolved_units: int = 0
    unresolved_rows: int = 0
    duplicate_rows: int = 0
    gold_source_labels: int = 0
    pivot_anchored_ids: int = 0
    units_by_hops: dict[str, int] = field(default_factory=dict)


def _match_key(text: str) -> str:
    """Match key for an English line.

    Args:
        text (str): The English line.

    Returns:
        str: A normalised key.
    """
    return " ".join(DIGIT_PATTERN.sub(" ", normalise(text)).split())


class GoldEnglishIndex:
    """Class for lookup over XED's human-annotated English lines, tolerant of NER anonymisation."""

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
            self.exact.setdefault(_match_key(text), label_ids)

            return

        segments = [_match_key(segment) for segment in PLACEHOLDER_PATTERN.split(text)]

        # A line that is nothing but a placeholder would match anything, so it is unusable.
        if not any(segments):
            return

        pattern = re.compile(".+?".join(re.escape(segment) for segment in segments))

        self.buckets[segments[0].split(" ")[0]].append((pattern, label_ids))

    def lookup(self, text: str) -> frozenset[int] | None:
        """Look up an English line.

        Args:
            text (str): The English line.

        Returns:
            frozenset[int] | None: A set of gold label ids, or None if the line is not annotated.
        """
        key = _match_key(text)

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


@dataclass(frozen=True)
class Bridge:
    """The English side reached from a Danish-referenced alignment."""

    doc: str
    sentence_id: str
    text: str
    english_id: bool


class PivotIndex:
    """Class for Danish-to-English bridges for annotations XED recorded against Danish subtitles."""

    def __init__(self) -> None:
        """Initialise the class."""
        self.phrases: dict[tuple[str, str], str] = {}
        self.links: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)

    def add_annotation(self, row: AnnotationRow) -> None:
        """Index one annotation recorded against a Danish sentence.

        Args:
            row (AnnotationRow): The annotation.
        """
        self.phrases.setdefault((row.doc, row.sentence_id), row.phrase)

    def add_link(self, row: PairRow) -> None:
        """Index one Danish-to-English alignment.

        Args:
            row (PairRow): The alignment, oriented with English as the target.
        """
        self.links[(row.source_doc, row.source_id)].append(
            (row.target_doc, row.target_id, row.target_text)
        )

    def lookup(self, row: PairRow) -> Bridge | None:
        """Find the annotated English line a Danish-sourced row bridges to.

        Args:
            row (PairRow): A target-language alignment whose source side is Danish.

        Returns:
            Bridge | None: The English side, or None if the Danish sentence was not annotated.
        """
        location = (row.source_doc, row.source_id)
        phrase = self.phrases.get(location)

        if phrase is None:
            return None

        key = _match_key(phrase)

        for doc, sentence_id, text in self.links.get(location, []):
            if _match_key(text) == key:
                return Bridge(doc, sentence_id, text, english_id=True)

        return Bridge(row.source_doc, row.source_id, phrase, english_id=False)


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
            gold = index.lookup(candidate.text)

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

    def _read_pivot(self) -> PivotIndex:
        """Index the Danish-to-English alignments that bridge pivot rows back to English.

        Returns:
            PivotIndex: A collection of bridges, keyed by Danish sentence.
        """
        pivot = PivotIndex()

        for annotation in self.corpus.iter_annotations(self.corpus.annotations_path):
            if annotation.doc.startswith(f"{PIVOT_LANG}/"):
                pivot.add_annotation(annotation)

        # Oriented with English as the target, so the source side is the pivot language.
        for row in self.corpus.iter_pairs(self.corpus.pivot_path, SOURCE_LANG):
            if row.source_doc.startswith(f"{PIVOT_LANG}/"):
                pivot.add_link(row)

        self.logger.info(
            f"Read {self.corpus.annotations_path.name}: {len(pivot.phrases)} annotations "
            f"recorded against {PIVOT_LANG} subtitles"
        )

        return pivot

    def _collect_candidates(self) -> dict[str, list[Candidate]]:
        """Collect every English line that aligns to each target-language line.

        Returns:
            dict[str, list[Candidate]]: A dictionary containing the alignment candidates per
                                        normalised target sentence.
        """
        candidates: dict[str, list[Candidate]] = defaultdict(list)
        pivot = self._read_pivot() if self.use_pivot else PivotIndex()

        for row in self.corpus.iter_pairs(self.corpus.pairs_path, self.corpus.target_lang):
            key = normalise(row.target_text)

            if row.source_doc.startswith(f"{SOURCE_LANG}/"):
                candidates[key].append(Candidate(DIRECT_HOPS, row.align_id(), row.source_text))
                self.stats.direct_alignments += 1

                continue

            bridge = pivot.lookup(row)

            if bridge is None:
                continue

            candidates[key].append(
                Candidate(BRIDGED_HOPS, row.align_id(bridge.doc, bridge.sentence_id), bridge.text)
            )
            self.stats.bridged_alignments += 1
            self.stats.pivot_anchored_ids += not bridge.english_id

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
