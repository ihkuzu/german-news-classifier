import pytest

from newsclf import baseline
from newsclf.data import split_dev


def test_baseline_learns_the_toy_topics(examples):
    model = baseline.train(examples, min_df=1)
    predictions = baseline.predict(
        model,
        ["Das Spiel im Stadion endet mit drei Toren", "Die Aktie fällt an der Börse"],
    )
    assert predictions == ["Sport", "Wirtschaft"]


def test_baseline_survives_save_and_load(tmp_path, examples):
    model = baseline.train(examples, min_df=1)
    path = tmp_path / "models" / "baseline.joblib"
    baseline.save(model, path)
    texts = [e.text for e in examples]
    assert baseline.predict(baseline.load(path), texts) == baseline.predict(model, texts)


def test_tune_scores_each_setting(examples):
    train, dev = split_dev(examples, fraction=0.25)
    rows = baseline.tune(train, dev, grid=[("svm", 1, 1.0), ("logreg", 2, 5.0)], min_df=1)
    assert [(r["kind"], r["ngrams"], r["c"]) for r in rows] == [("svm", 1, 1.0), ("logreg", 2, 5.0)]
    assert all(r["dev_accuracy"] == 1.0 for r in rows)


def test_unknown_classifier_is_rejected():
    with pytest.raises(ValueError, match="unknown classifier"):
        baseline.build(kind="forest")
