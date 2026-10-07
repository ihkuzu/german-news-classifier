import pytest

from newsclf import cli
from newsclf.report import load_all


def run(tmp_path, data_dir, *argv):
    base = ["--data", str(data_dir), "--results", str(tmp_path / "results")]
    base += ["--models", str(tmp_path / "models")]
    return cli.main(base + list(argv))


@pytest.fixture()
def small_baseline(monkeypatch):
    from newsclf import baseline

    original = baseline.build
    # the toy corpus is too small for the default min_df
    monkeypatch.setattr(baseline, "build", lambda **kwargs: original(min_df=1, **kwargs))


def test_baseline_writes_result_and_model(tmp_path, data_dir, capsys, small_baseline):
    assert run(tmp_path, data_dir, "baseline") == 0
    assert "| tfidf-svm | 9 | 1.000 |" in capsys.readouterr().out
    assert (tmp_path / "models" / "baseline.joblib").exists()
    [result] = load_all(tmp_path / "results")
    assert result.indices is None
    assert len(result.predictions) == 9
    assert result.details == {"train_articles": 24}


def test_predict_uses_the_saved_baseline(tmp_path, data_dir, capsys, small_baseline):
    run(tmp_path, data_dir, "baseline")
    capsys.readouterr()
    assert run(tmp_path, data_dir, "predict", "Tore im Stadion beim Spiel") == 0
    assert capsys.readouterr().out.strip() == "Sport"


def test_predict_without_a_model_explains_what_to_do(tmp_path, data_dir, capsys):
    assert run(tmp_path, data_dir, "predict", "x") == 2
    assert "run the baseline command" in capsys.readouterr().err


def test_report_and_errors(tmp_path, data_dir, capsys, small_baseline):
    run(tmp_path, data_dir, "baseline")
    capsys.readouterr()
    assert run(tmp_path, data_dir, "report") == 0
    out = capsys.readouterr().out
    assert "Full test set" in out
    assert "| Sport | 1.00 | 1.00 | 1.00 | 3 |" in out
    assert run(tmp_path, data_dir, "errors", "tfidf-svm") == 0
    assert run(tmp_path, data_dir, "errors", "missing") == 2
    assert "available: tfidf-svm" in capsys.readouterr().err


def test_llm_scores_a_sample(tmp_path, data_dir, capsys, monkeypatch):
    from newsclf import llm

    class FakeClassifier:
        def __init__(self, model):
            self.model = model

        def classify(self, text):
            return "Sport" if "Stadion" in text else None

    monkeypatch.setattr(llm, "OllamaClassifier", FakeClassifier)
    assert run(tmp_path, data_dir, "llm", "--model", "tiny:1b", "-n", "6") == 0
    [result] = load_all(tmp_path / "results")
    assert result.name == "zero-shot-tiny:1b"
    assert len(result.indices) == 6
    assert result.predictions.count("Sport") == 2
    assert result.predictions.count(None) == 4
    assert "| zero-shot-tiny:1b | 6 | 0.333 |" in capsys.readouterr().out


def test_llm_stops_when_the_server_never_answers(tmp_path, data_dir, capsys, monkeypatch):
    from newsclf import llm

    class DeadClassifier:
        def __init__(self, model):
            self.calls = 0

        def classify(self, text):
            raise llm.LLMError("Ollama request failed: refused")

    monkeypatch.setattr(llm, "OllamaClassifier", DeadClassifier)
    assert run(tmp_path, data_dir, "llm", "-n", "9") == 1
    assert "refused" in capsys.readouterr().err
    assert load_all(tmp_path / "results") == []


def test_tune_prints_one_row_per_setting(tmp_path, data_dir, capsys, monkeypatch):
    from newsclf import baseline

    monkeypatch.setattr(baseline, "GRID", [("svm", 1, 1.0), ("logreg", 2, 5.0)])
    original = baseline.tune
    monkeypatch.setattr(
        baseline, "tune", lambda train, dev: original(train, dev, baseline.GRID, min_df=1)
    )
    assert run(tmp_path, data_dir, "tune") == 0
    rows = capsys.readouterr().out.splitlines()[2:]
    assert [r.split("|")[1:4] for r in rows] == [[" svm ", " 1-1 ", " 1 "], [" logreg ", " 1-2 ", " 5 "]]


def test_download_lists_existing_files(tmp_path, data_dir, capsys):
    assert run(tmp_path, data_dir, "download") == 0
    assert capsys.readouterr().out.split() == [
        str(data_dir / "train.csv"),
        str(data_dir / "test.csv"),
    ]
