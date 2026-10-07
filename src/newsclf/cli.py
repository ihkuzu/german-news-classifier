from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import baseline, report
from .data import download, load_split, sample_indices, split_dev
from .report import Result


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def cmd_download(args) -> int:
    for path in download(args.data):
        print(path)
    return 0


def cmd_baseline(args) -> int:
    train = load_split(args.data, "train")
    test = load_split(args.data, "test")
    started = time.perf_counter()
    model = baseline.train(train)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = baseline.predict(model, [e.text for e in test])
    predict_seconds = time.perf_counter() - started
    baseline.save(model, Path(args.models) / "baseline.joblib")
    result = Result(
        name="tfidf-svm",
        method="TF-IDF + linear SVM",
        gold=[e.label for e in test],
        predictions=predictions,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        details={"train_articles": len(train)},
    )
    report.save(result, args.results)
    print(report.format_report([result]))
    return 0


def cmd_tune(args) -> int:
    train, dev = split_dev(load_split(args.data, "train"), seed=args.seed)
    print("| classifier | n-grams | C | dev accuracy |")
    print("| --- | --- | --- | --- |")
    for row in baseline.tune(train, dev):
        print(f"| {row['kind']} | 1-{row['ngrams']} | {row['c']:g} | {row['dev_accuracy']:.3f} |")
    return 0


def cmd_finetune(args) -> int:
    from . import transformer

    train, dev = split_dev(load_split(args.data, "train"), seed=args.seed)
    test = load_split(args.data, "test")
    name = args.name or args.model.split("/")[-1]
    out_dir = Path(args.models) / report.safe_name(name)

    def on_epoch(entry):
        _log(
            f"epoch {entry['epoch']}  loss {entry['loss']:.3f}  "
            f"dev accuracy {entry['dev_accuracy']:.3f}  {entry['seconds']:.0f} s"
        )

    started = time.perf_counter()
    history = transformer.finetune(
        train,
        dev,
        out_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        seed=args.seed,
        on_epoch=on_epoch,
    )
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = transformer.predict(out_dir, [e.text for e in test], max_length=args.max_length)
    predict_seconds = time.perf_counter() - started
    result = Result(
        name=name,
        method=f"fine-tuned {args.model}",
        gold=[e.label for e in test],
        predictions=predictions,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        details={
            "train_articles": len(train),
            "dev_articles": len(dev),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(transformer._device()),
            "history": history,
        },
    )
    report.save(result, args.results)
    print(report.format_report([result]))
    return 0


def cmd_llm(args) -> int:
    from .llm import LLMError, OllamaClassifier

    test = load_split(args.data, "test")
    indices = sample_indices(test, args.n, seed=args.seed)
    classifier = OllamaClassifier(model=args.model)
    predictions: list[str | None] = []
    failures = 0
    started = time.perf_counter()
    for done, index in enumerate(indices, start=1):
        try:
            predictions.append(classifier.classify(test[index].text))
        except LLMError as exc:
            failures += 1
            predictions.append(None)
            # nothing answered yet, so the server is probably not running
            if failures == done >= 3:
                _log(str(exc))
                return 1
        if done % 25 == 0:
            _log(f"{done}/{len(indices)}")
    predict_seconds = time.perf_counter() - started
    result = Result(
        name=f"zero-shot-{args.model}",
        method=f"zero-shot prompt, {args.model} (Ollama)",
        gold=[test[i].label for i in indices],
        predictions=predictions,
        indices=indices,
        predict_seconds=predict_seconds,
        details={"failed_requests": failures, "seed": args.seed},
    )
    report.save(result, args.results)
    print(report.format_report([result]))
    return 0


def cmd_report(args) -> int:
    results = report.load_all(args.results)
    print(report.format_report(results))
    repeats = report.format_repeats(results)
    if repeats:
        print(f"\n{repeats}")
    for result in results:
        print(f"\n{result.name}: {result.method}\n")
        print(report.format_classes(result))
    return 0


def cmd_errors(args) -> int:
    results = {r.name: r for r in report.load_all(args.results)}
    if args.name not in results:
        _log(f"no result named {args.name!r}, available: {', '.join(results) or 'none'}")
        return 2
    result = results[args.name]
    test = load_split(args.data, "test")
    indices = result.indices or list(range(len(test)))
    shown = 0
    for index, gold, pred in zip(indices, result.gold, result.predictions):
        if gold == pred:
            continue
        print(f"[{index}] true {gold}, predicted {pred or '(none)'}")
        print(f"    {test[index].text[:args.chars]}")
        shown += 1
        if shown >= args.k:
            break
    return 0


def cmd_predict(args) -> int:
    path = Path(args.models) / "baseline.joblib"
    if args.name == "baseline":
        if not path.exists():
            _log(f"{path} not found, run the baseline command first")
            return 2
        print(baseline.predict(baseline.load(path), [args.text])[0])
        return 0
    from . import transformer

    print(transformer.predict(Path(args.models) / report.safe_name(args.name), [args.text])[0])
    return 0


def main(argv: list[str] | None = None) -> int:
    # German text must survive a redirect to a file on Windows
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="newsclf")
    parser.add_argument("--data", default="data", help="folder with train.csv and test.csv")
    parser.add_argument("--results", default="results", help="folder for result files")
    parser.add_argument("--models", default="models", help="folder for trained models")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch the dataset").set_defaults(func=cmd_download)
    sub.add_parser("baseline", help="train and score TF-IDF + linear SVM").set_defaults(
        func=cmd_baseline
    )

    p = sub.add_parser("tune", help="compare baseline settings on a dev split of the training set")
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("finetune", help="fine-tune a BERT model and score it")
    p.add_argument("--model", default="deepset/gbert-base")
    p.add_argument("--name", help="name for the result, defaults to the model name")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("llm", help="score a zero-shot prompt on a sample of the test set")
    p.add_argument("--model", default="llama3.2:3b")
    p.add_argument("-n", type=int, default=300, help="number of test articles")
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_llm)

    sub.add_parser("report", help="compare all saved results").set_defaults(func=cmd_report)

    p = sub.add_parser("errors", help="show misclassified articles of one result")
    p.add_argument("name")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--chars", type=int, default=300)
    p.set_defaults(func=cmd_errors)

    p = sub.add_parser("predict", help="classify one text")
    p.add_argument("text")
    p.add_argument("--name", default="baseline", help="baseline or a fine-tuned result name")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)
