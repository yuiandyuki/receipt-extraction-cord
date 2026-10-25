"""Print the README results tables from results/ (so every number in the README has a source).

    python -m receipt.report
"""

import argparse
import json
from pathlib import Path

from .baseline import FIELD as TOTAL_FIELD

MAIN_FIELDS = [
    "menu.nm", "menu.cnt", "menu.price", "menu.unitprice",
    "sub_total.subtotal_price", "sub_total.tax_price",
    TOTAL_FIELD, "total.cashprice", "total.changeprice",
]


def fmt_ci(s: dict, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(s['point'])} [{f.format(s['ci95'][0])}, {f.format(s['ci95'][1])}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    r = args.results_dir

    text = json.loads((r / "metrics" / f"text_seed{args.seed}.json").read_text())
    layout = json.loads((r / "metrics" / f"layout_seed{args.seed}.json").read_text())
    baseline = json.loads((r / "metrics" / "baseline.json").read_text())
    stats = json.loads((r / "stats.json").read_text())["comparisons"]

    print(f"Test set (100 receipts), entity-level F1, seed {args.seed}. 95% CI: receipt-level bootstrap.\n")
    print("| Field | Support | Rule baseline | Text (XLM-R) | Layout (LayoutLMv3) |")
    print("|---|---:|---:|---:|---:|")
    micro = stats["layout_vs_text/micro"]
    total = stats["layout_vs_text/" + TOTAL_FIELD]
    base_total = stats[f"text_vs_baseline/{TOTAL_FIELD}"]["a"]
    print(f"| **micro (all fields)** | {int(text['test']['micro']['support'])} | – | {fmt_ci(micro['a'])} | {fmt_ci(micro['b'])} |")
    for field in MAIN_FIELDS:
        t = text["test"]["per_field"].get(field, {})
        l = layout["test"]["per_field"].get(field, {})
        if field == TOTAL_FIELD:
            row = f"| `{field}` | {int(t.get('support', 0))} | {fmt_ci(base_total)} | {fmt_ci(total['a'])} | {fmt_ci(total['b'])} |"
        else:
            row = f"| `{field}` | {int(t.get('support', 0))} | – | {t.get('f1', 0):.3f} | {l.get('f1', 0):.3f} |"
        print(row)

    print("\nPaired comparisons on test (difference of micro F1, 95% bootstrap CI, two-sided permutation p):\n")
    print("| Comparison | Scope | Difference | p |")
    print("|---|---|---:|---:|")
    names = {"layout_vs_text": "Layout − Text", "text_vs_baseline": "Text − Baseline", "layout_vs_baseline": "Layout − Baseline"}
    for key, c in stats.items():
        pair, scope = key.split("/", 1)
        print(f"| {names[pair]} | {scope} | {fmt_ci(c['b_minus_a'], signed=True)} | {c['permutation']['p_value']:.4f} |")

    print("\nTraining runs:\n")
    print("| Run | Device | Train time | Best epoch | Validation micro F1 | Test micro F1 |")
    print("|---|---|---:|---:|---:|---:|")
    for path in sorted((r / "metrics").glob("*_seed*.json")):
        m = json.loads(path.read_text())
        print(
            f"| {m['run']} | {m['machine']['device']} | {m['train_seconds'] / 60:.1f} min | {m['best_epoch']:.0f} | "
            f"{m['validation']['micro']['f1']:.4f} | {m['test']['micro']['f1']:.4f} |"
        )


if __name__ == "__main__":
    main()
