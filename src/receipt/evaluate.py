"""Prediction back to word level, and entity-level metrics with seqeval."""

import json
from pathlib import Path

import numpy as np
import torch

from .bio import bio_entities
from .data import LABELS
from .encode import encode


def is_layout_model(model) -> bool:
    return model.config.model_type == "layoutlmv3"


@torch.no_grad()
def predict(model, tokenizer, receipts, batch_size: int = 16) -> list[list[str]]:
    """Word-level BIO tags for each receipt. Each word takes the prediction of its first
    sub-token; words cut off by truncation are tagged O (and so count as misses)."""
    model.eval()
    device = next(model.parameters()).device
    use_layout = is_layout_model(model)
    out = []
    for start in range(0, len(receipts), batch_size):
        batch = receipts[start:start + batch_size]
        enc, word_ids = encode(
            tokenizer,
            [r.words for r in batch],
            boxes=[r.norm_boxes for r in batch] if use_layout else None,
            pixel_values=[r.pixel_values for r in batch] if use_layout else None,
        )
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        pred_ids = logits.argmax(-1).cpu().numpy()
        for b, r in enumerate(batch):
            tags = ["O"] * len(r.words)
            seen = set()
            for pos, wid in enumerate(word_ids[b]):
                if wid is not None and wid not in seen:
                    tags[wid] = model.config.id2label[int(pred_ids[b, pos])]
                    seen.add(wid)
            out.append(tags)
    return out


def entity_metrics(gold: list[list[str]], pred: list[list[str]]) -> dict:
    """Entity-level precision / recall / F1: micro average over all fields, plus per field.
    An entity counts as correct only if both its field and its exact word span match."""
    from seqeval.metrics import classification_report

    report = classification_report(gold, pred, output_dict=True, zero_division=0)
    micro = report["micro avg"]
    per_field = {
        name: {k.replace("-score", ""): float(v) for k, v in scores.items()}
        for name, scores in report.items()
        if not name.endswith(" avg")
    }
    return {
        "micro": {k.replace("-score", ""): float(v) for k, v in micro.items()},
        "per_field": dict(sorted(per_field.items())),
    }


def receipt_counts(gold: list[str], pred: list[str], field: str | None = None) -> tuple[int, int, int]:
    """(TP, FP, FN) entity counts for one receipt, optionally restricted to one field.
    Summing these over receipts gives exactly seqeval's micro precision / recall / F1."""
    g = set(bio_entities(gold))
    p = set(bio_entities(pred))
    if field is not None:
        g = {e for e in g if e[0] == field}
        p = {e for e in p if e[0] == field}
    tp = len(g & p)
    return tp, len(p) - tp, len(g) - tp


def compute_metrics_fn(eval_pred):
    """For the HF Trainer (checkpoint selection on validation). Positions with label -100 are
    sub-tokens / special tokens / padding; the rest are words in order."""
    from seqeval.metrics import f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    gold, pred = [], []
    for p_row, l_row in zip(preds, labels):
        mask = l_row != -100
        gold.append([LABELS[i] for i in l_row[mask]])
        pred.append([LABELS[i] for i in p_row[mask]])
    return {"f1": f1_score(gold, pred, zero_division=0)}


def save_predictions(path: Path, receipts, pred_tags: list[list[str]]):
    rows = [
        {"image_id": r.image_id, "words": r.words, "gold": r.tags, "pred": p}
        for r, p in zip(receipts, pred_tags)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False) + "\n")


def load_predictions(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())
