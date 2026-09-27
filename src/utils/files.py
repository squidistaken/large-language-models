import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1 << 20


def checksum(path: Path) -> str:
    """Compute the SHA-256 of a file.

    Args:
        path (Path): The file to hash.

    Returns:
        str: A hex digest.
    """
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSON Lines, one compact object per line.

    Args:
        path (Path): The file to write.
        rows (list[dict]): The rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    """Write an indented JSON document.

    Args:
        path (Path): The file to write.
        payload (Any): The document to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON document.

    Args:
        path (Path): The file to read.

    Returns:
        Any: The parsed document.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON Lines file.

    Args:
        path (Path): The file to read.

    Returns:
        list[dict]: A list of parsed rows.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
