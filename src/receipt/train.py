"""Fine-tune one model on CORD-v2 train, choose the best epoch on validation, evaluate on test once.

    python -m receipt.train --model text   --seed 42
    python -m receipt.train --model layout --seed 42

Writes the best checkpoint to runs/<model>_seed<seed>/best/ (not in git), metrics to
results/metrics/<model>_seed<seed>.json and word-level predictions to results/predictions/.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoImageProcessor,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .data import LABEL2ID, LABELS, load_split
from .encode import MAX_LENGTH, Collator, ReceiptDataset
from .evaluate import compute_metrics_fn, entity_metrics, predict, save_predictions

MODELS = {
    "text": "FacebookAI/xlm-roberta-base",  # words only
    "layout": "microsoft/layoutlmv3-base",  # words + boxes + image
}


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if torch.cuda.is_available():
        info["device"] = "cuda: " + torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            chip = platform.processor()
        info["device"] = f"mps: {chip}"
    else:
        info["device"] = "cpu: " + platform.processor()
    return info


def token_lengths(tokenizer, receipts, use_layout: bool) -> list[int]:
    lengths = []
    for r in receipts:
        if use_layout:
            enc = tokenizer(r.words, boxes=r.norm_boxes, truncation=False)
        else:
            enc = tokenizer(r.words, is_split_into_words=True, truncation=False)
        lengths.append(len(enc["input_ids"]))
    return lengths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N receipts per split (smoke test)")
    args = ap.parse_args()

    run_name = f"{args.model}_seed{args.seed}"
    run_dir = args.runs_dir / run_name
    use_layout = args.model == "layout"
    checkpoint = MODELS[args.model]

    set_seed(args.seed)  # also seeds the randomly initialised classification head
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, apply_ocr=False) if use_layout else None
    splits = {s: load_split(s, image_processor) for s in ("train", "validation", "test")}
    if args.limit:
        splits = {s: rs[: args.limit] for s, rs in splits.items()}
    lengths = {s: token_lengths(tokenizer, rs, use_layout) for s, rs in splits.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )

    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.1,  # a float in [0, 1) is a ratio of total steps
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        seed=args.seed,
        data_seed=args.seed,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,  # the collator needs words / boxes / pixel_values
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=Collator(tokenizer, use_layout),
        train_dataset=ReceiptDataset(splits["train"]),
        eval_dataset=ReceiptDataset(splits["validation"]),
        compute_metrics=compute_metrics_fn,
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start

    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    if image_processor is not None:
        image_processor.save_pretrained(str(best_dir))

    # Model selection used validation only; test is evaluated exactly once, here.
    model = trainer.model
    results = {}
    for split in ("validation", "test"):
        receipts = splits[split]
        preds = predict(model, tokenizer, receipts)
        save_predictions(args.results_dir / "predictions" / f"{run_name}_{split}.json", receipts, preds)
        results[split] = entity_metrics([r.tags for r in receipts], preds)

    val_curve = [
        {"epoch": h["epoch"], "f1": h["eval_f1"]} for h in trainer.state.log_history if "eval_f1" in h
    ]
    metrics = {
        "run": run_name,
        "checkpoint": checkpoint,
        "command": "python -m receipt.train " + " ".join(sys.argv[1:]),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "scheduler": "linear",
            "max_length": MAX_LENGTH,
            "seed": args.seed,
            "fp16": targs.fp16,
        },
        "machine": machine_info(),
        "train_seconds": round(train_seconds, 1),
        "best_epoch": next(
            (c["epoch"] for c in val_curve if c["f1"] == trainer.state.best_metric), None
        ),
        "best_validation_f1_during_training": trainer.state.best_metric,
        "validation_f1_per_epoch": val_curve,
        "max_tokens": {s: max(v) for s, v in lengths.items()},
        "receipts_truncated": {s: sum(n > MAX_LENGTH for n in v) for s, v in lengths.items()},
        "limit": args.limit,
        **results,
    }
    out = args.results_dir / "metrics" / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"run": run_name, "test_micro": results["test"]["micro"], "train_seconds": metrics["train_seconds"]}, indent=2))
    print(f"wrote {out}; best model in {best_dir}")


if __name__ == "__main__":
    main()
