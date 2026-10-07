from collections import Counter

import pytest

from newsclf.data import (
    FILES,
    Example,
    download,
    load_split,
    read_csv,
    sample_indices,
    split_dev,
)


def test_read_csv_handles_quotes_and_separators(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text(
        "Sport;Tor in der 90. Minute\n"
        "Web;'Apple; Google und ''Meta'' im Test'\n",
        encoding="utf-8",
    )
    assert read_csv(path) == [
        Example("Sport", "Tor in der 90. Minute"),
        Example("Web", "Apple; Google und 'Meta' im Test"),
    ]


def test_read_csv_keeps_umlauts(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Kultur;Größe, Würde und Öl\n", encoding="utf-8")
    assert read_csv(path)[0].text == "Größe, Würde und Öl"


def test_read_csv_rejects_unknown_label(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Wetter;Regen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        read_csv(path)


def test_read_csv_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Sport;a;b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2 columns"):
        read_csv(path)


def test_download_fetches_missing_files_only(tmp_path):
    (tmp_path / "train.csv").write_text("Sport;alt\n", encoding="utf-8")
    calls = []

    def fetch(url):
        calls.append(url)
        return "Sport;neu\n".encode()

    paths = download(tmp_path, base_url="http://x/", fetch=fetch)
    assert [p.name for p in paths] == list(FILES)
    assert calls == ["http://x/test.csv"]
    assert (tmp_path / "train.csv").read_text(encoding="utf-8") == "Sport;alt\n"


def test_load_split_says_how_to_get_the_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_split(tmp_path, "train")


def test_split_dev_is_disjoint_and_stratified(examples):
    train, dev = split_dev(examples, fraction=0.25)
    assert len(train) + len(dev) == len(examples)
    assert not set(train) & set(dev)
    assert Counter(e.label for e in dev) == {"Sport": 2, "Wirtschaft": 2, "Web": 2}


def test_split_dev_is_repeatable(examples):
    assert split_dev(examples, seed=5) == split_dev(examples, seed=5)
    assert split_dev(examples, seed=5) != split_dev(examples, seed=6)


def test_sample_indices_keeps_proportions(examples):
    indices = sample_indices(examples, 6)
    assert indices == sorted(set(indices))
    assert Counter(examples[i].label for i in indices) == {"Sport": 2, "Wirtschaft": 2, "Web": 2}


def test_sample_indices_returns_everything_when_n_is_large(examples):
    assert sample_indices(examples, 1000) == list(range(len(examples)))
