"""Receipt-level bootstrap confidence intervals and a paired permutation test for micro F1.

Words (and entities) on the same receipt are not independent: one hard-to-read receipt makes
many errors at once. So the unit we resample / permute is the receipt, not the word.

Usage:
    python -m receipt.stats \
        --system text=results/predictions/text_seed42_test.json \
        --system layout=results/predictions/layout_seed42_test.json \
        --system baseline=results/predictions/baseline_test.json \
        --out results/stats.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .baseline import FIELD as TOTAL_FIELD
from .evaluate import load_predictions, receipt_counts


def micro_f1(counts: np.ndarray) -> np.ndarray:
    """counts[..., 0:3] = TP, FP, FN (already summed over receipts). F1 = 2TP / (2TP + FP + FN)."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros(np.shape(denom), dtype=float), where=denom > 0)


def counts_matrix(rows: list[dict], field: str | None = None) -> np.ndarray:
    """(n_receipts, 3) array of per-receipt TP / FP / FN."""
    return np.array([receipt_counts(r["gold"], r["pred"], field) for r in rows], dtype=np.int64)


def bootstrap(counts_a: np.ndarray, counts_b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Resample receipts with replacement (the same draw for both systems, since they are scored
    on the same receipts), re-sum TP/FP/FN, recompute micro F1. Percentile 95% intervals."""
    rng = np.random.default_rng(seed)
    n = len(counts_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    f1_a = micro_f1(counts_a[idx].sum(axis=1))
    f1_b = micro_f1(counts_b[idx].sum(axis=1))

    def summary(point, samples):
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return {"point": float(point), "ci95": [float(lo), float(hi)]}

    point_a = micro_f1(counts_a.sum(axis=0))
    point_b = micro_f1(counts_b.sum(axis=0))
    return {
        "a": summary(point_a, f1_a),
        "b": summary(point_b, f1_b),
        "b_minus_a": summary(point_b - point_a, f1_b - f1_a),
    }


def permutation_test(counts_a: np.ndarray, counts_b: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> dict:
    """Paired permutation test. H0: the two systems are exchangeable, i.e. on every receipt it is
    a coin flip which system produced which prediction. Under H0 we swap the two systems'
    predictions on each receipt with probability 0.5 and recompute the F1 difference.
    Two-sided p = share of permuted |diff| at least as large as the observed |diff|."""
    rng = np.random.default_rng(seed)
    n = len(counts_a)
    observed = float(micro_f1(counts_b.sum(axis=0)) - micro_f1(counts_a.sum(axis=0)))
    swap = rng.random((n_perm, n)) < 0.5
    perm_a = np.where(swap[..., None], counts_b, counts_a).sum(axis=1)
    perm_b = np.where(swap[..., None], counts_a, counts_b).sum(axis=1)
    diffs = micro_f1(perm_b) - micro_f1(perm_a)
    # +1 in numerator and denominator: the observed assignment is one of the permutations.
    extreme = np.sum(np.abs(diffs) >= abs(observed) - 1e-12)
    p_value = (extreme + 1) / (n_perm + 1)
    return {"observed_diff": observed, "p_value": float(p_value), "n_perm": n_perm}


def compare(rows_a, rows_b, field=None, n_boot=10_000, n_perm=10_000, seed=0) -> dict:
    ids_a = [r["image_id"] for r in rows_a]
    ids_b = [r["image_id"] for r in rows_b]
    if ids_a != ids_b:
        raise ValueError("the two prediction files must cover the same receipts in the same order")
    ca, cb = counts_matrix(rows_a, field), counts_matrix(rows_b, field)
    out = bootstrap(ca, cb, n_boot, seed)
    out["permutation"] = permutation_test(ca, cb, n_perm, seed)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="name=path/to/predictions.json")
    ap.add_argument("--out", type=Path, default=Path("results/stats.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    systems = dict(s.split("=", 1) for s in args.system)
    rows = {name: load_predictions(path) for name, path in systems.items()}
    kw = dict(n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)

    result = {
        "unit": "receipt",
        "n_receipts": len(next(iter(rows.values()))),
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "inputs": systems,
        "comparisons": {},
    }
    if "text" in rows and "layout" in rows:
        result["comparisons"]["layout_vs_text/micro"] = compare(rows["text"], rows["layout"], None, **kw)
        result["comparisons"]["layout_vs_text/" + TOTAL_FIELD] = compare(rows["text"], rows["layout"], TOTAL_FIELD, **kw)
    if "baseline" in rows:
        for name in ("text", "layout"):
            if name in rows:
                result["comparisons"][f"{name}_vs_baseline/{TOTAL_FIELD}"] = compare(
                    rows["baseline"], rows[name], TOTAL_FIELD, **kw
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for key, c in result["comparisons"].items():
        a, b, d = c["a"], c["b"], c["b_minus_a"]
        print(
            f"{key}: a={a['point']:.4f} [{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]  "
            f"b={b['point']:.4f} [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]  "
            f"b-a={d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
            f"p={c['permutation']['p_value']:.4f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
