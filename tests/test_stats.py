import random

import numpy as np
from seqeval.metrics import f1_score
from seqeval.metrics.sequence_labeling import get_entities

from receipt.bio import bio_entities
from receipt.evaluate import receipt_counts
from receipt.stats import bootstrap, micro_f1, permutation_test


def random_tags(rng, n):
    return [rng.choice(["O", "B-a", "I-a", "B-b", "I-b"]) for _ in range(n)]


def test_entity_counts_match_seqeval():
    rng = random.Random(0)
    gold = [random_tags(rng, rng.randint(1, 12)) for _ in range(200)]
    pred = [random_tags(rng, len(g)) for g in gold]
    for g in gold:
        assert bio_entities(g) == get_entities(g)
    # Summing per-receipt TP/FP/FN reproduces seqeval's micro F1 (what the bootstrap relies on).
    counts = np.array([receipt_counts(g, p) for g, p in zip(gold, pred)])
    assert np.isclose(micro_f1(counts.sum(axis=0)), f1_score(gold, pred))


def test_identical_systems_show_no_difference():
    rng = np.random.default_rng(1)
    tp = rng.integers(5, 20, size=100)
    counts = np.stack([tp, rng.integers(0, 3, size=100), rng.integers(0, 3, size=100)], axis=1)
    boot = bootstrap(counts, counts, n_boot=2000)
    assert boot["b_minus_a"]["point"] == 0.0
    assert boot["b_minus_a"]["ci95"] == [0.0, 0.0]
    lo, hi = boot["a"]["ci95"]
    assert lo <= boot["a"]["point"] <= hi
    assert permutation_test(counts, counts, n_perm=2000)["p_value"] == 1.0


def simulate(rng, n_receipts, hit_rate):
    """Each receipt has 10 gold entities; the system finds each with prob. hit_rate and
    makes one wrong prediction for every miss. Returns (n_receipts, 3) TP/FP/FN."""
    tp = rng.binomial(10, hit_rate, size=n_receipts)
    return np.stack([tp, 10 - tp, 10 - tp], axis=1)


def test_clearly_better_system_is_detected():
    rng = np.random.default_rng(2)
    a, b = simulate(rng, 100, 0.6), simulate(rng, 100, 0.9)
    boot = bootstrap(a, b, n_boot=2000)
    assert 0.5 < boot["a"]["point"] < 0.7 and 0.85 < boot["b"]["point"] < 0.95
    assert boot["b_minus_a"]["ci95"][0] > 0
    assert permutation_test(a, b, n_perm=2000)["p_value"] < 0.01


def test_equal_systems_are_not_called_different():
    rng = np.random.default_rng(3)
    a, b = simulate(rng, 100, 0.8), simulate(rng, 100, 0.8)
    boot = bootstrap(a, b, n_boot=2000)
    lo, hi = boot["b_minus_a"]["ci95"]
    assert lo < 0 < hi
    assert permutation_test(a, b, n_perm=2000)["p_value"] > 0.05
