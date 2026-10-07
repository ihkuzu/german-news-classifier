import pytest

from newsclf.data import Example

TOPICS = {
    "Sport": "Der Verein gewinnt das Spiel im Stadion mit zwei Toren",
    "Wirtschaft": "Die Bank meldet Gewinn und die Aktie steigt an der Börse",
    "Web": "Das neue Smartphone bekommt ein Update für die App",
}


def make_examples(per_label: int = 8) -> list[Example]:
    examples = []
    for label, sentence in TOPICS.items():
        for i in range(per_label):
            examples.append(Example(label, f"{sentence}, Meldung Nummer {i}."))
    return examples


def write_csv(path, examples) -> None:
    lines = []
    for e in examples:
        text = e.text.replace("'", "''")
        lines.append(f"{e.label};'{text}'" if ";" in e.text or "'" in e.text else f"{e.label};{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def examples():
    return make_examples()


@pytest.fixture()
def data_dir(tmp_path, examples):
    folder = tmp_path / "data"
    folder.mkdir()
    write_csv(folder / "train.csv", examples)
    write_csv(folder / "test.csv", make_examples(per_label=3))
    return folder
