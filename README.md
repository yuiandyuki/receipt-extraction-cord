# Receipt field extraction on CORD-v2: does layout help?

[![CI](https://github.com/yuiandyuki/receipt-extraction-cord/actions/workflows/ci.yml/badge.svg)](https://github.com/yuiandyuki/receipt-extraction-cord/actions/workflows/ci.yml)

Given the OCR words of a receipt (text + box per word), tag which words are the item names, prices,
subtotal, tax, total, cash, change, ... — the step after OCR in an AI-OCR bookkeeping flow, where a
photographed receipt becomes a journal-entry candidate. I compare a **text-only** model
(`xlm-roberta-base`, sees only the words) with a **layout-aware** model (`microsoft/layoutlmv3-base`,
sees words + coordinates + the image) and a **regex rule baseline** for the total, and test whether the
difference is statistically meaningful on a 100-receipt test set.

> Status: code, tests, CI and Docker are done. **Results below are filled in after training**
> (`python -m receipt.report` prints the tables from `results/`).

## Results

<!-- Paste the output of `python -m receipt.report` here after training. -->

_TBD after training._

**What the results say** <!-- TODO(Sihao): write this part in your own words. -->

_TBD._

### An error example

_TBD after training: one test receipt, its words, gold vs predicted tags, and why the model got it wrong._

## How it works

```
CORD-v2 ground_truth (JSON)                    src/receipt/data.py
  valid_line[i].category, words[j].text/quad/is_key
        │  one line = one entity; B- on first word, I- on the rest; key words ("TOTAL") -> O
        ▼
word-level sequence: words, pixel boxes, BIO tags (59 labels = O + B/I x 29 fields)
        │  boxes -> 0-1000 grid (layout model only); image -> 224x224 pixel_values
        ▼
sub-word tokens (XLM-R / LayoutLMv3 tokenizer)  src/receipt/encode.py
        │  only the first sub-token of each word carries the label, the rest -100
        ▼
token classification fine-tuning (HF Trainer)   src/receipt/train.py
        │  each word takes its first sub-token's prediction
        ▼
word-level predicted tags -> entities           src/receipt/evaluate.py, bio.py
        │  seqeval: an entity is correct only if field AND exact word span match
        ▼
micro / per-field P, R, F1; receipt-level bootstrap + permutation test   src/receipt/stats.py
```

### Data

[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2): Indonesian restaurant/shop receipts,
800 train / 100 validation / 100 test. Each annotated line has a category (e.g. `menu.nm`,
`total.total_price`) and words with a quadrilateral box. I use the dataset's own OCR words and boxes.

What the data looked like versus the plan (checked by printing samples before writing the converter):

- The structure matched (`valid_line` → `category` + `words` with `text` / `quad`; size in `meta.image_size`).
- **Key words are inside the value's line.** A `total.total_price` line is e.g. `Grand(key) Total(key) 1,591,600`.
  I tag leading/trailing key words `O`, so the entity is the amount itself, which is what bookkeeping needs
  and what the rule baseline outputs. (`include_keys=True` gives the conventional labelling; numbers are
  therefore not directly comparable with published CORD F1 scores.)
- 29 categories occur (all present in train); 19 words in total have boxes slightly outside the image, so boxes are clipped.
- Receipts are short: median 20 words, max 135. See [`notebooks/eda.ipynb`](notebooks/eda.ipynb) for field counts —
  three menu fields make up most entities, and several fields have fewer than 10 training examples.

### Models and training

| | Text model | Layout model |
|---|---|---|
| Checkpoint | `FacebookAI/xlm-roberta-base` (278M params, MIT) | `microsoft/layoutlmv3-base` (133M params, CC-BY-NC-SA 4.0) |
| Input | words | words + boxes (0–1000) + 224×224 image |
| Settings | 15 epochs, batch 8, AdamW lr 5e-5, weight decay 0.01, 10% linear warm-up, max 512 tokens, seed 42 | same |

The best epoch is picked by validation micro F1; the test set is evaluated exactly once, after training.

### Statistics

- **Metric**: entity-level micro F1 (seqeval, BIO, default mode). A field value is useful only if the whole
  value is right, so a token-level score would overstate quality (and would be dominated by `O`).
- **Bootstrap by receipt**: resample the 100 test receipts with replacement 10,000 times; each time re-sum TP / FP / FN
  over the sampled receipts and recompute micro F1 → percentile 95% CI for each model and for *layout − text*.
- **Paired permutation test**: H0 = the two models are interchangeable. On each receipt, swap the two models'
  predictions with probability 0.5, recompute the F1 difference; 10,000 times; two-sided p.
- **Why receipts, not words**: words on one receipt are not independent (a blurry or unusual receipt causes many
  errors at once), so resampling words would pretend we have thousands of independent observations and give
  intervals that are far too narrow. The receipt is the independent unit.
- With only 100 test receipts the intervals are wide; a small F1 gap may not be distinguishable from noise.

## Design decisions

- **Use the dataset's OCR** rather than training OCR: the question here is field extraction given OCR output;
  OCR quality is a separate problem (and a known limitation below).
- **Same training settings for both models**, so the comparison is about the input (layout or not), not tuning.
- **First-sub-token labelling**: one prediction per word, which is what the API returns.
- **Rule baseline only for `total.total_price`**: the one field a simple rule can target (keyword + amount);
  it shows what the models add over a few lines of regex. The rule was written by looking at train/validation only.
- **Keys tagged `O`**: see Data.

## Known limitations

- Test set is small (100 receipts): wide confidence intervals; rare fields have a handful of test entities.
- Only Indonesian restaurant/shop receipts; no Japanese receipts, no invoices.
- Uses the dataset's own (clean) OCR. With a real OCR engine, word errors and **reading order** would differ:
  words here come in annotation order, grouped by line, which is easier than real OCR output — probably more so
  for the text-only model, which has no other way to know which words belong together.
- One seed per model unless noted; the variance across seeds is not measured.

## Reproduce

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate   # or python3.11 -m venv .venv
pip install -r requirements-dev.txt && pip install --no-deps -e .
pytest -q

python -m receipt.train --model text   --seed 42   # -> results/metrics/text_seed42.json, runs/text_seed42/best/
python -m receipt.train --model layout --seed 42
python -m receipt.baseline                          # -> results/metrics/baseline.json
python -m receipt.stats \
  --system text=results/predictions/text_seed42_test.json \
  --system layout=results/predictions/layout_seed42_test.json \
  --system baseline=results/predictions/baseline_test.json \
  --out results/stats.json
python -m receipt.report                            # README tables
```

On Google Colab: open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) (GPU runtime) and run all cells.
Machine, library versions and training time of each run are recorded in `results/metrics/*.json`.

## API

`POST /parse`, multipart: `image` (the receipt image) + `ocr` (JSON `{"words": [...], "boxes": [[x0, y0, x1, y1], ...]}`
in pixels of that image). Returns the extracted fields with their word indices.

```bash
docker build -t receipt-api .
docker run --rm -v "$(pwd)/model:/app/model" -p 8000:8000 receipt-api
curl -s -F image=@examples/receipt.jpg -F "ocr=<examples/ocr.json" localhost:8000/parse
```

_Example output: TBD after training._

### Getting `model/`

Weights are not in git. Either retrain (`python -m receipt.train ...`, then `cp -r runs/<run>/best model`)
or download `model-*.zip` from the GitHub Release and unzip it to `model/`.

## Repository layout

```
src/receipt/   data.py (CORD -> BIO)  encode.py (tokenise/align)  train.py  evaluate.py  bio.py
               baseline.py (regex)  stats.py (bootstrap/permutation)  report.py  api.py (FastAPI)
notebooks/     eda.ipynb  train_colab.ipynb
tests/         BIO conversion, boxes, statistics on synthetic data, API with a tiny random model
results/       metrics/*.json, predictions/*.json, stats.json
examples/      one CORD validation receipt for the API demo
```

## Licenses

- **Data**: CORD-v2, CC-BY-4.0 — Park et al., *CORD: A Consolidated Receipt Dataset for Post-OCR Parsing*, Document
  Intelligence Workshop at NeurIPS 2019. `examples/` and `results/predictions/` contain CORD words and one CORD image.
- **LayoutLMv3 weights are CC-BY-NC-SA 4.0 (non-commercial)**, so the fine-tuned layout model may not be used
  commercially; a product would need a commercially licensed layout model. XLM-RoBERTa is MIT.
