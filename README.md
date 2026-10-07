# german-news-classifier

[![ci](https://github.com/ihkuzu/german-news-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ihkuzu/german-news-classifier/actions/workflows/ci.yml)

Topic classification of German news articles, done three ways and scored on the same
test set:

1. a classic baseline: TF-IDF features and a linear SVM
2. a fine-tuned German BERT model
3. a zero-shot prompt to a small local LLM

The question I wanted to answer is the practical one: how much accuracy does each
step up in model size buy, and what does it cost in training time and time per
article?

## Data

[10kGNAD](https://tblock.github.io/10kGNAD/) has 10,273 articles from the Austrian
newspaper Der Standard in nine sections: Etat (media), Inland, International, Kultur,
Panorama, Sport, Web, Wirtschaft and Wissenschaft. It comes with a fixed split of
9,245 training and 1,028 test articles, which I use unchanged so the numbers can be
compared with other work on this dataset.

The data is licensed CC BY-NC-SA 4.0 and is not part of this repository.
`python -m newsclf download` fetches it into `data/`.

## Results

Measured on one desktop PC (RTX 2060 with 6 GB, one run per method).

Full test set (1,028 articles):

| method | accuracy | macro F1 | training | per article |
| --- | --- | --- | --- | --- |
| TF-IDF + linear SVM | 0.887 | 0.892 | 4 s (CPU) | 0.3 ms |
| fine-tuned gbert-base | 0.907 | 0.901 | 442 s (GPU) | 4.5 ms |

The same 301 test articles for all three (the LLM is too slow for the full set):

| method | accuracy | macro F1 | per article |
| --- | --- | --- | --- |
| TF-IDF + linear SVM | 0.894 | 0.900 | 0.3 ms |
| fine-tuned gbert-base | 0.920 | 0.913 | 4.5 ms |
| zero-shot llama3.2:3b | 0.482 | 0.493 | 481 ms |

What I take from this:

- **Fine-tuning wins, but by two points.** BERT gets 932 articles right, the baseline
  912. It costs about 100 times the training time, a GPU and 15 times the time per
  article. Whether that is worth it depends on what a wrong label costs.
- **The gap is at the edge of what this test set can show.** BERT is right on 59
  articles where the baseline is wrong, the baseline on 39 where BERT is wrong (sign
  test p = 0.054). With a single training run I would not call that settled.
- **The comparison slightly favours the baseline.** It is trained on all 9,245
  training articles, while BERT gives up 924 of them as its dev split.
- **A small general model with a prompt is far behind.** It labels almost half of the
  articles wrongly and is the slowest by a wide margin. Its favourite mistake is
  Inland: precision 0.31 there, because articles from Panorama, International and
  Wirtschaft get filed under domestic politics. The section names follow one
  newspaper's habits, which a prompt cannot learn from nine one-line descriptions.
- **Where the methods differ.** BERT is clearly better on Web (F1 0.98 against 0.91)
  and Wirtschaft (0.90 against 0.86). The baseline is better on Kultur (0.90 against
  0.85) and Wissenschaft (0.94 against 0.91), the two smallest sections. Both share
  the same top mistake, International filed under Panorama, 12 times each.

BERT's dev accuracy went 0.856, 0.895, 0.897 over the three epochs, so more epochs
would probably add little. Every number above comes from the files in `results/`,
and `python -m newsclf report` prints the full per-section tables.

## How the comparison is kept fair

- **The test set is only used for the final score.** Baseline settings were chosen
  on a dev split (10% of the training articles, stratified), and the fine-tuning run
  uses the same split to pick its best epoch.
- **Same articles for every method.** The LLM is slow, so it is scored on a
  stratified sample of the test set. The report then scores every other method on
  exactly those articles as well, next to the full-set numbers.
- **An unusable LLM reply counts as wrong.** If the model returns a label that is not
  one of the nine sections, that article is a miss, and the report says how many
  there were.
- **Timing is part of the result.** Each run records training time and time per
  article on the machine it ran on.

Choosing the baseline on the dev split (`python -m newsclf tune`):

| classifier | n-grams | C | dev accuracy |
| --- | --- | --- | --- |
| svm | 1-1 | 0.3 | 0.864 |
| svm | 1-1 | 1 | 0.865 |
| svm | 1-1 | 3 | 0.865 |
| logreg | 1-1 | 5 | 0.863 |
| logreg | 1-1 | 20 | 0.860 |
| logreg | 1-1 | 80 | 0.864 |
| svm | 1-2 | 0.3 | 0.851 |
| svm | 1-2 | 1 | 0.859 |
| svm | 1-2 | 3 | 0.858 |
| logreg | 1-2 | 5 | 0.842 |
| logreg | 1-2 | 20 | 0.851 |
| logreg | 1-2 | 80 | 0.850 |

With 924 dev articles, differences below about one point are noise. Word pairs did
not help, and the SVM and logistic regression are level, so I kept the simplest and
fastest option: single words and a linear SVM with C=1.

## Run it

```bash
pip install -e .
python -m newsclf download
python -m newsclf baseline
python -m newsclf predict "Die Europäische Zentralbank senkt den Leitzins."
```

Fine-tuning needs PyTorch and a GPU. The default settings ran on a 6 GB card:

```bash
pip install -e ".[train]"
python -m newsclf finetune                      # deepset/gbert-base, 3 epochs
python -m newsclf finetune --model distilbert-base-german-cased
```

The zero-shot run needs [Ollama](https://ollama.com) with a model pulled:

```bash
python -m newsclf llm --model llama3.2:3b -n 300
```

Then compare everything that has been run, or look at mistakes of one method:

```bash
python -m newsclf report
python -m newsclf errors tfidf-svm -k 5
```

Every run writes a JSON file to `results/` with its predictions, timings and
settings. The report is built from those files, so a number in the table can always
be traced back to a run.

## Design notes

- **Fine-tuning loop.** Plain PyTorch instead of a trainer class: AdamW, linear
  warm-up and decay, gradient clipping and mixed precision on the GPU. After each
  epoch the model is scored on the dev split and only the best epoch is kept.
- **Article length.** Articles are about 2,600 characters on average. BERT reads the
  first 256 tokens by default (`--max-length`) to keep memory use low. News articles
  state their topic early, so this loses less than it sounds, but it is a trade-off.
- **Prompt.** The LLM gets a one-line description of each section, because names such
  as Etat and Panorama do not explain themselves, and only the first 1,500
  characters of the article. It has to answer with a JSON object.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

53 tests. They need neither the dataset nor a model download: the CSV reader, the
metrics (checked against values worked out by hand), the baseline on a toy corpus,
the Ollama client against mocked HTTP responses, the report and the command line.
The fine-tuning loop is tested by training a tiny, randomly initialised BERT on a toy
corpus until it fits. Those tests are skipped when PyTorch is not installed, and CI
runs them in a separate job.

## Limits

- One dataset from one newspaper. Section labels reflect that paper's editorial
  habits, so the numbers say little about other sources.
- One training run per model, no repeats with different seeds yet.
- The section an article was published in is not always the only reasonable label,
  which puts a ceiling on accuracy for every method.

## Roadmap

- [x] Dataset download, fixed train/test split
- [x] TF-IDF baseline, settings chosen on a dev split
- [x] Fine-tuning and zero-shot code with tests
- [x] Measure the fine-tuned German BERT
- [x] Measure the zero-shot LLM on a test sample
- [ ] Few-shot prompt and a larger LLM
- [ ] Compare a multilingual BERT with the German one
- [ ] Repeat the fine-tuning with several seeds

## License

MIT for the code. The dataset has its own license, see above.
