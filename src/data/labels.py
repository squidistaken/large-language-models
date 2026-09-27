from dataclasses import dataclass

from src.constants import EMOTIONS


@dataclass(frozen=True)
class LabelSpace:
    """The multi-label space."""

    names: tuple[str, ...] = EMOTIONS

    def __len__(self) -> int:
        """Count the labels.

        Returns:
            int: A number of labels.
        """
        return len(self.names)

    @property
    def ids(self) -> range:
        """The one-based label ids, in order.

        Returns:
            range: A range of label ids.
        """
        return range(1, len(self.names) + 1)

    def name(self, label_id: int) -> str:
        """Look up a label name by id.

        Args:
            label_id (int): The one-based label id.

        Returns:
            str: A label name.
        """
        return self.names[label_id - 1]

    def parse_ids(self, raw: str) -> frozenset[int]:
        """Parse a XED label field into a set of label ids.

        Args:
            raw (str): The raw comma-separated label field.

        Returns:
            frozenset[int]: A set of parsed label ids.
        """
        ids = set()

        for part in raw.split(","):
            part = part.strip()

            if part.isdigit() and int(part) in self.ids:
                ids.add(int(part))

        return frozenset(ids)

    def to_names(self, label_ids: frozenset[int]) -> list[str]:
        """Convert label ids to label names in the fixed order.

        Args:
            label_ids (frozenset[int]): The label ids.

        Returns:
            list[str]: A list of label names.
        """
        return [name for index, name in enumerate(self.names, start=1) if index in label_ids]

    def to_vector(self, label_ids: frozenset[int]) -> list[int]:
        """Binarise label ids against the fixed order.

        Args:
            label_ids (frozenset[int]): The label ids.

        Returns:
            list[int]: A binary vector, one element per label.
        """
        return [int(label_id in label_ids) for label_id in self.ids]


XED_LABELS = LabelSpace()
