from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from .data import Example
from .metrics import accuracy

# every setting that tune() tries: classifier, largest n-gram, C
GRID = [
    (kind, ngrams, c)
    for ngrams in (1, 2)
    for kind, values in (("svm", (0.3, 1.0, 3.0)), ("logreg", (5.0, 20.0, 80.0)))
    for c in values
]


def build(kind: str = "svm", ngrams: int = 1, c: float = 1.0, min_df: int = 2) -> Pipeline:
    if kind == "svm":
        classifier = LinearSVC(C=c, random_state=0)
    elif kind == "logreg":
        classifier = LogisticRegression(C=c, max_iter=1000)
    else:
        raise ValueError(f"unknown classifier {kind!r}")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, ngrams),
        sublinear_tf=True,
        min_df=min_df,
        max_features=200_000,
    )
    return make_pipeline(vectorizer, classifier)


def train(examples: list[Example], **kwargs) -> Pipeline:
    model = build(**kwargs)
    model.fit([e.text for e in examples], [e.label for e in examples])
    return model


def predict(model: Pipeline, texts: list[str]) -> list[str]:
    return [str(label) for label in model.predict(texts)]


def tune(train_set: list[Example], dev: list[Example], grid=GRID, **kwargs) -> list[dict]:
    rows = []
    gold = [e.label for e in dev]
    for kind, ngrams, c in grid:
        model = train(train_set, kind=kind, ngrams=ngrams, c=c, **kwargs)
        score = accuracy(gold, predict(model, [e.text for e in dev]))
        rows.append({"kind": kind, "ngrams": ngrams, "c": c, "dev_accuracy": score})
    return rows


def save(model: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)
