from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import LABELS
from .metrics import accuracy, macro_f1, per_class, top_confusions


@dataclass
class Result:
    name: str
    method: str
    gold: list[str]
    predictions: list[str | None]
    # positions in the test set, None means the whole set in order
    indices: list[int] | None = None
    train_seconds: float | None = None
    predict_seconds: float | None = None
    details: dict = field(default_factory=dict)

    def restricted(self, indices: list[int]) -> tuple[list[str], list[str | None]]:
        if self.indices is None:
            return [self.gold[i] for i in indices], [self.predictions[i] for i in indices]
        position = {index: i for i, index in enumerate(self.indices)}
        rows = [position[i] for i in indices]
        return [self.gold[i] for i in rows], [self.predictions[i] for i in rows]

    def covers(self, indices: list[int]) -> bool:
        return self.indices is None or set(indices) <= set(self.indices)


def safe_name(text: str) -> str:
    # model names contain ":" and "/", which Windows file names cannot
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def save(result: Result, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe_name(result.name)}.json"
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_all(folder: str | Path) -> list[Result]:
    paths = sorted(Path(folder).glob("*.json"))
    return [Result(**json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def _seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} s" if value >= 10 else f"{value:.1f} s"


def _row(result: Result, gold, pred) -> str:
    speed = "-"
    if result.predict_seconds is not None and result.predictions:
        speed = f"{1000 * result.predict_seconds / len(result.predictions):.1f} ms"
    return (
        f"| {result.name} | {len(gold)} | {accuracy(gold, pred):.3f} "
        f"| {macro_f1(gold, pred, LABELS):.3f} | {_seconds(result.train_seconds)} | {speed} |"
    )


HEADER = (
    "| method | articles | accuracy | macro F1 | training | per article |\n"
    "| --- | --- | --- | --- | --- | --- |"
)


def format_report(results: list[Result]) -> str:
    if not results:
        return "no results yet"
    lines = []
    full = [r for r in results if r.indices is None]
    if full:
        lines += ["Full test set", "", HEADER]
        lines += [_row(r, r.gold, r.predictions) for r in full]
    for partial in (r for r in results if r.indices is not None):
        # every method that also covers these articles, scored on the same ones
        lines += ["", f"Same {len(partial.indices)} articles as {partial.name}", "", HEADER]
        for other in results:
            if other.covers(partial.indices):
                lines.append(_row(other, *other.restricted(partial.indices)))
    return "\n".join(lines)


def format_repeats(results: list[Result]) -> str:
    # runs of the same method that differ only in their seed
    groups: dict[str, list[Result]] = {}
    for result in results:
        if result.indices is None:
            groups.setdefault(result.method, []).append(result)
    lines = []
    for method, runs in groups.items():
        if len(runs) < 2:
            continue
        scores = [accuracy(r.gold, r.predictions) for r in runs]
        mean = sum(scores) / len(scores)
        spread = (sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5
        lines.append(
            f"| {method} | {len(runs)} | {mean:.3f} | {spread:.3f} "
            f"| {min(scores):.3f} | {max(scores):.3f} |"
        )
    if not lines:
        return ""
    header = [
        "Repeated runs",
        "",
        "| method | runs | mean accuracy | std | min | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    return "\n".join(header + lines)


def format_classes(result: Result) -> str:
    scores = per_class(result.gold, result.predictions, LABELS)
    lines = [
        "| section | precision | recall | F1 | articles |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, s in scores.items():
        lines.append(
            f"| {label} | {s['precision']:.2f} | {s['recall']:.2f} "
            f"| {s['f1']:.2f} | {s['support']} |"
        )
    lines.append("")
    lines.append("most frequent mistakes (true section -> predicted):")
    for gold, pred, count in top_confusions(result.gold, result.predictions):
        lines.append(f"  {gold} -> {pred}: {count}")
    unanswered = sum(p is None for p in result.predictions)
    if unanswered:
        lines.append(f"no valid label returned: {unanswered}")
    return "\n".join(lines)
