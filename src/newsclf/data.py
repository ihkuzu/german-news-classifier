from __future__ import annotations

import csv
import random
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/tblock/10kGNAD/master/"
FILES = ("train.csv", "test.csv")
LABELS = (
    "Etat",
    "Inland",
    "International",
    "Kultur",
    "Panorama",
    "Sport",
    "Web",
    "Wirtschaft",
    "Wissenschaft",
)


@dataclass(frozen=True)
class Example:
    label: str
    text: str


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def download(folder: str | Path, base_url: str = BASE_URL, fetch=_fetch) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        target = folder / name
        if not target.exists():
            target.write_bytes(fetch(base_url + name))
        paths.append(target)
    return paths


def read_csv(path: str | Path) -> list[Example]:
    # articles are long, the default field limit is too small
    csv.field_size_limit(10_000_000)
    examples = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter=";", quotechar="'"):
            if len(row) != 2:
                raise ValueError(f"{path}: expected 2 columns, got {len(row)}")
            label, text = row
            if label not in LABELS:
                raise ValueError(f"{path}: unknown label {label!r}")
            examples.append(Example(label, text))
    return examples


def load_split(folder: str | Path, name: str) -> list[Example]:
    path = Path(folder) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run the download command first")
    return read_csv(path)


def _by_label(examples: list[Example]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.label].append(index)
    return groups


def split_dev(
    examples: list[Example], fraction: float = 0.1, seed: int = 13
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    held_out: set[int] = set()
    groups = _by_label(examples)
    for label in sorted(groups):
        indices = groups[label]
        rng.shuffle(indices)
        held_out.update(indices[: max(1, round(len(indices) * fraction))])
    train = [e for i, e in enumerate(examples) if i not in held_out]
    dev = [e for i, e in enumerate(examples) if i in held_out]
    return train, dev


def sample_indices(examples: list[Example], n: int, seed: int = 13) -> list[int]:
    # keeps the class proportions of the full set
    if n >= len(examples):
        return list(range(len(examples)))
    rng = random.Random(seed)
    chosen: list[int] = []
    groups = _by_label(examples)
    for label in sorted(groups):
        indices = groups[label]
        rng.shuffle(indices)
        chosen.extend(indices[: max(1, round(n * len(indices) / len(examples)))])
    return sorted(chosen)
