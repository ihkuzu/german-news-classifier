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
