"""Exact re-implementation of the leaderboard metric: per-Source-1 F0.5, macro-averaged."""


def f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    # F_beta = (1+b^2) TP / (b^2 |truth| + |pred|)
    return 1.25 * tp / (0.25 * len(truth) + len(pred))


def macro_f05(pred: dict, truth: dict, ids=None) -> float:
    ids = list(truth) if ids is None else list(ids)
    if not ids:
        return float("nan")
    return sum(f05(set(pred.get(i, ())), truth.get(i, set())) for i in ids) / len(ids)


def blocking_report(cands: dict, truth: dict, n_right: int) -> dict:
    """Recall ceiling of a candidate set, and how much it shrinks the all-pairs space."""
    n_true = sum(len(v) for v in truth.values())
    found = sum(len(truth[i] & set(cands.get(i, ()))) for i in truth)
    n_pairs = sum(len(cands.get(i, ())) for i in truth)
    ceiling = macro_f05({i: truth[i] & set(cands.get(i, ())) for i in truth}, truth)
    return {
        "pair_recall": found / max(n_true, 1),
        "f05_ceiling": ceiling,
        "cands_per_s1": n_pairs / max(len(truth), 1),
        "reduction_ratio": 1 - n_pairs / max(len(truth) * n_right, 1),
    }
