from __future__ import annotations

import json
import random
import time
from pathlib import Path

from .data import LABELS, Example
from .metrics import accuracy

DEFAULT_MODEL = "deepset/gbert-base"


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(model_name: str | Path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except ValueError:
        # some older BERT repos ship only vocab.txt, which the auto class can miss
        from huggingface_hub import hf_hub_download
        from transformers import BertTokenizerFast

        config_path = hf_hub_download(str(model_name), "tokenizer_config.json")
        settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return BertTokenizerFast(
            hf_hub_download(str(model_name), "vocab.txt"),
            do_lower_case=settings.get("do_lower_case", True),
            strip_accents=settings.get("strip_accents"),
        )


def _load_model(model_name: str | Path, **kwargs):
    from transformers import AutoModelForSequenceClassification

    try:
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    except ValueError:
        # the same older repos have no model_type in config.json
        from transformers import BertForSequenceClassification

        return BertForSequenceClassification.from_pretrained(model_name, **kwargs)


def _batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _encode(tokenizer, texts: list[str], max_length: int, device):
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _predict_ids(model, tokenizer, texts, batch_size, max_length, device) -> list[int]:
    import torch

    model.eval()
    # similar lengths in one batch means less padding
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    ids = [0] * len(texts)
    with torch.no_grad():
        for chunk in _batches(order, batch_size):
            inputs = _encode(tokenizer, [texts[i] for i in chunk], max_length, device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**inputs).logits
            for index, label_id in zip(chunk, logits.argmax(dim=-1).tolist()):
                ids[index] = label_id
    return ids


def finetune(
    train: list[Example],
    dev: list[Example],
    out_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 256,
    lr: float = 3e-5,
    seed: int = 13,
    on_epoch=None,
) -> list[dict]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    device = _device()
    label_ids = {label: i for i, label in enumerate(LABELS)}

    tokenizer = _load_tokenizer(model_name)
    model = _load_model(
        model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=label_ids,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = []
    best = -1.0
    order = list(range(len(train)))
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        random.shuffle(order)
        total = 0.0
        for chunk in _batches(order, batch_size):
            inputs = _encode(tokenizer, [train[i].text for i in chunk], max_length, device)
            labels = torch.tensor([label_ids[train[i].label] for i in chunk], device=device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = model(**inputs, labels=labels).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(chunk)

        ids = _predict_ids(
            model, tokenizer, [e.text for e in dev], batch_size * 2, max_length, device
        )
        dev_accuracy = accuracy([e.label for e in dev], [LABELS[i] for i in ids])
        entry = {
            "epoch": epoch,
            "loss": total / len(train),
            "dev_accuracy": dev_accuracy,
            "seconds": time.perf_counter() - started,
        }
        history.append(entry)
        if on_epoch:
            on_epoch(entry)
        # keep the epoch that did best on the dev set, not the last one
        if dev_accuracy > best:
            best = dev_accuracy
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    return history


def predict(
    model_dir: str | Path, texts: list[str], batch_size: int = 32, max_length: int = 256
) -> list[str]:
    device = _device()
    tokenizer = _load_tokenizer(model_dir)
    model = _load_model(model_dir).to(device)
    ids = _predict_ids(model, tokenizer, texts, batch_size, max_length, device)
    return [model.config.id2label[i] for i in ids]
