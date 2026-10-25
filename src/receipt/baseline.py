"""Rule baseline: regex + keywords, extracts only total.total_price.

Rule: find a "total"-type keyword (TOTAL, GRAND TOTAL, DUE, ...) that is not a subtotal or an
item/quantity count, take the first amount-looking word within the next few words, and among
all such candidates keep the one with the largest amount (the grand total is normally the
largest of the totals). The rule was written by looking at train/validation receipts only.
"""

import re

FIELD = "total.total_price"

TOTAL_KEYS = {"total", "grandtotal", "ttl", "tl", "due", "du", "amount", "totalbelanja"}
NOT_AFTER = {"sub"}  # "SUB TOTAL"
NOT_BEFORE = {"qty", "item", "items", "quantity", "pcs", "disc", "discount", "tax", "menu"}
AMOUNT = re.compile(r"^(?:rp\.?)?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?$|^(?:rp\.?)?\d+(?:[.,]\d{1,2})?$", re.I)
LOOKAHEAD = 3


def _norm(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def _amount_value(word: str) -> float:
    digits = re.sub(r"[^\d]", "", word)
    return float(digits) if digits else 0.0


def predict_total(words: list[str]) -> list[str]:
    tags = ["O"] * len(words)
    candidates = []
    for i, word in enumerate(words):
        key = _norm(word)
        if key not in TOTAL_KEYS and not key.endswith("total"):
            continue
        if key.startswith("sub") or (i > 0 and _norm(words[i - 1]) in NOT_AFTER):
            continue
        if i + 1 < len(words) and _norm(words[i + 1]) in NOT_BEFORE:
            continue
        for j in range(i + 1, min(i + 1 + LOOKAHEAD, len(words))):
            if AMOUNT.match(words[j]):
                candidates.append((_amount_value(words[j]), j))
                break
    if candidates:
        _, j = max(candidates)
        tags[j] = "B-" + FIELD
    return tags


def predict(receipts) -> list[list[str]]:
    return [predict_total(r.words) for r in receipts]


def main():
    """python -m receipt.baseline  ->  results/metrics/baseline.json, results/predictions/baseline_*.json"""
    import argparse
    import json
    from pathlib import Path

    from .data import load_split
    from .evaluate import receipt_counts, save_predictions

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    metrics = {"field": FIELD, "note": "scores only the total.total_price field"}
    for split in ("train", "validation", "test"):
        receipts = load_split(split)
        preds = predict(receipts)
        if split != "train":
            save_predictions(args.results_dir / "predictions" / f"baseline_{split}.json", receipts, preds)
        tp = fp = fn = 0
        for r, p in zip(receipts, preds):
            a, b, c = receipt_counts(r.tags, p, FIELD)
            tp, fp, fn = tp + a, fp + b, fn + c
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        metrics[split] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        print(f"{split}: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    out = args.results_dir / "metrics" / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
