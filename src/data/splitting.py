from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.constants import DEFAULT_SEED, DEFAULT_SPLITS, SPLIT_NAMES
from src.data.alignment import AlignmentUnit
from src.data.labels import XED_LABELS, LabelSpace
from src.utils.logger import Logger
from src.utils.text import normalise

STRATIFIER = "iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit"


def validate_splits(splits: tuple[float, ...] | list[float]) -> None:
    """Check that split ratios can partition the data.

    Args:
        splits (tuple[float, ...] | list[float]): The train, development and test fractions.

    Raises:
        ValueError: If there are not three ratios, any is not positive, or they do not sum to one.
    """
    if len(splits) != len(SPLIT_NAMES):
        raise ValueError(f"Provide {len(SPLIT_NAMES)} split ratios, got {len(splits)}.")

    if any(share <= 0 for share in splits):
        raise ValueError(f"Every split ratio must be greater than 0, got {list(splits)}.")

    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(splits):g}.")


@dataclass
class SplitStats:
    """Counts describing how units were deduplicated, grouped and partitioned."""

    duplicate_rows: int = 0
    leakage_groups: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)


class SplitAssigner:
    """Assigns aligned pairs to train, development and test partitions."""

    def __init__(
        self,
        logger: Logger,
        splits: tuple[float, ...] = DEFAULT_SPLITS,
        seed: int = DEFAULT_SEED,
        labels: LabelSpace = XED_LABELS,
    ) -> None:
        """Initialise the class.

        Args:
            logger (Logger): The logger.
            splits (tuple[float, ...]): The train, development and test fractions.
            seed (int): The seed fixing the split.
            labels (LabelSpace): The label space to stratify over.

        Raises:
            ValueError: If the fractions are not three values summing to one.
        """
        validate_splits(splits)

        self.logger = logger
        self.splits = splits
        self.seed = seed
        self.labels = labels
        self.stats = SplitStats()

    def deduplicate(self, units: list[AlignmentUnit]) -> list[AlignmentUnit]:
        """Drop pairs that repeat an existing pair on both sides.

        Args:
            units (list[AlignmentUnit]): The resolved alignment units.

        Returns:
            list[AlignmentUnit]: A list of deduplicated units, ordered by alignment identifier.
        """
        unique = {}

        for unit in sorted(units, key=lambda item: item.align_id):
            key = (normalise(unit.source_text), normalise(unit.target_text))

            if key in unique:
                self.stats.duplicate_rows += 1

                continue

            unique[key] = unit

        self.logger.info(f"Dropped {self.stats.duplicate_rows} further exact duplicate pair(s)")

        return list(unique.values())

    def assign(self, units: list[AlignmentUnit]) -> list[str]:
        """Assign every unit to a partition using multi-label iterative stratification.

        Args:
            units (list[AlignmentUnit]): The deduplicated units.

        Returns:
            list[str]: A list of partition names for each unit.
        """
        groups = self._group(units)
        labels = np.array([self._group_vector(units, group) for group in groups])
        features = np.arange(len(groups)).reshape(-1, 1)
        train, held_out = self._partition(features, labels, sum(self.splits[1:]))
        test_share = self.splits[2] / sum(self.splits[1:])
        dev, test = self._partition(features[held_out], labels[held_out], test_share)
        assignments = [""] * len(units)

        for name, group_indices in zip(
            SPLIT_NAMES, (train, held_out[dev], held_out[test]), strict=True
        ):
            for group_index in group_indices:
                for unit_index in groups[group_index]:
                    assignments[unit_index] = name

        self.stats.split_sizes = dict(Counter(assignments))
        self.logger.info(f"Split sizes: {self.stats.split_sizes}")

        return assignments

    def _group(self, units: list[AlignmentUnit]) -> list[list[int]]:
        """Get group units that share an English line.

        Args:
            units (list[AlignmentUnit]): The deduplicated units.

        Returns:
            list[list[int]]: A list of index groups, each of which must land in a single partition.

        Raises:
            ValueError: If a target line occurs in more than one unit.
        """
        repeated = [
            text
            for text, count in Counter(normalise(unit.target_text) for unit in units).items()
            if count > 1
        ]

        if repeated:
            raise ValueError(
                f"{len(repeated)} target line(s) occur in more than one unit, so grouping by "
                "English text alone would let them span partitions."
            )

        grouped = defaultdict(list)

        for index, unit in enumerate(units):
            grouped[normalise(unit.source_text)].append(index)

        groups = list(grouped.values())
        self.stats.leakage_groups = sum(1 for group in groups if len(group) > 1)
        self.logger.info(
            f"{len(groups)} partition groups "
            f"({self.stats.leakage_groups} contain more than one alignment unit)"
        )

        return groups

    def _group_vector(self, units: list[AlignmentUnit], group: list[int]) -> list[int]:
        """Binarise the union of a group's target labels.

        Args:
            units (list[AlignmentUnit]): The deduplicated units.
            group (list[int]): The unit indices making up the group.

        Returns:
            list[int]: A list of group's binary label vector.
        """
        merged = frozenset().union(*(units[index].target_labels for index in group))

        return self.labels.to_vector(merged)

    def _partition(
        self, features: np.ndarray, labels: np.ndarray, minority_share: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split rows into a majority and a minority fold by iterative stratification.

        Args:
            features (np.ndarray): The placeholder features, one row per group.
            labels (np.ndarray): The binary label matrix, one row per group.
            minority_share (float): The fraction of rows to place in the minority fold.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing majority and minority row indices.
        """
        stratifier = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=minority_share, random_state=self.seed
        )
        majority, minority = next(stratifier.split(features, labels))

        return np.sort(majority), np.sort(minority)
