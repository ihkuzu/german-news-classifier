import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from newsclf import transformer
from newsclf.data import LABELS, split_dev

from conftest import make_examples


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    # a random two-layer BERT, so the test needs no download
    folder = tmp_path_factory.mktemp("tiny-bert")
    words = sorted({w.strip(",.").lower() for e in make_examples() for w in e.text.split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", "."] + words
    (folder / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    transformers.BertTokenizerFast(str(folder / "vocab.txt")).save_pretrained(folder)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    transformers.BertModel(config).save_pretrained(folder)
    return str(folder)


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_bert):
    folder = tmp_path_factory.mktemp("trained")
    train, dev = split_dev(make_examples(per_label=12), fraction=0.25)
    seen = []
    history = transformer.finetune(
        train,
        dev,
        folder,
        model_name=tiny_bert,
        epochs=20,
        batch_size=8,
        max_length=32,
        lr=5e-3,
        on_epoch=seen.append,
    )
    return folder, dev, history, seen


def test_finetune_learns_and_reports_each_epoch(trained):
    _, _, history, seen = trained
    assert [h["epoch"] for h in history] == list(range(1, 21))
    assert seen == history
    assert history[-1]["loss"] < history[0]["loss"]
    assert max(h["dev_accuracy"] for h in history) == 1.0


def test_saved_model_is_the_best_epoch(trained):
    folder, dev, _, _ = trained
    predictions = transformer.predict(folder, [e.text for e in dev], max_length=32)
    assert predictions == [e.label for e in dev]
    assert set(predictions) <= set(LABELS)


def test_prediction_order_does_not_depend_on_text_length(trained):
    folder, dev, _, _ = trained
    texts = [e.text for e in dev]
    mixed = [texts[0], texts[-1] + " " + texts[-1], texts[4]]
    together = transformer.predict(folder, mixed, batch_size=2, max_length=32)
    assert together == [dev[0].label, dev[-1].label, dev[4].label]
    assert len(set(together)) == 3


def test_tokenizer_falls_back_to_the_vocab_file(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / "vocab.txt").write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Größe", "zeigt", "##e"]) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        '{"do_lower_case": false, "strip_accents": false}', encoding="utf-8"
    )

    def broken(name):
        raise ValueError("Couldn't instantiate the backend tokenizer")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", broken)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, filename: str(tmp_path / filename)
    )
    tokenizer = transformer._load_tokenizer("some/model")
    # cased, and umlauts are not stripped
    assert tokenizer.tokenize("Größe zeigte") == ["Größe", "zeigt", "##e"]
    assert tokenizer.tokenize("größe") == ["[UNK]"]


def test_model_without_model_type_is_loaded_as_bert(tmp_path, tiny_bert):
    import json
    import shutil

    folder = tmp_path / "old-style"
    shutil.copytree(tiny_bert, folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    del config["model_type"]
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="model_type"):
        transformers.AutoModelForSequenceClassification.from_pretrained(folder)
    model = transformer._load_model(folder, num_labels=len(LABELS))
    assert model.config.model_type == "bert"
    assert model.config.num_labels == len(LABELS)

    train, dev = split_dev(make_examples(per_label=4), fraction=0.25)
    out = tmp_path / "out"
    transformer.finetune(train, dev, out, model_name=str(folder), epochs=1, batch_size=8, max_length=32)
    # the saved copy has the model type, so it loads the normal way
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["model_type"] == "bert"
    assert len(transformer.predict(out, [e.text for e in dev], max_length=32)) == len(dev)
