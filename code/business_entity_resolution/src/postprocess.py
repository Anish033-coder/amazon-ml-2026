"""Turning pair probabilities into per-Source-1 match lists that score well on
macro F0.5. The metric is per S1 entity, so the decision is made per S1 group."""
import itertools

import numpy as np
import pandas as pd

from .metric import macro_f05


def one_owner(scored: pd.DataFrame) -> pd.DataFrame:
    """Source 1 is deduplicated, so a S2/S3 record belongs to at most one S1 entity:
    keep only its highest-probability S1."""
    best = scored.groupby("r")["p"].idxmax()
    return scored.loc[best.to_numpy()]


def _groups(scored: pd.DataFrame):
    s = scored.sort_values(["s1", "p"], ascending=[True, False])
    s1, r, p = s["s1"].to_numpy(), s["r"].to_numpy(), s["p"].to_numpy()
    cuts = np.nonzero(s1[1:] != s1[:-1])[0] + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(s1)]):
        yield s1[a], r[a:b], p[a:b]


def select_threshold(scored, t: float, rel: float) -> dict:
    out = {}
    for sid, r, p in _groups(scored):
        keep = (p >= t) & (p >= rel * p[0])
        if keep.any():
            out[sid] = list(r[keep])
    return out


def _pb_pmf(ps):
    """Poisson-binomial pmf: distribution of the number of true matches among ps."""
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1.0
    for q in ps:
        pmf[1:] = pmf[1:] * (1 - q) + pmf[:-1] * q
        pmf[0] *= 1 - q
    return pmf


def best_k_expected_f05(ps, max_k: int = 8) -> int:
    """ps sorted descending. Returns k maximising E[F0.5] of predicting the top k,
    treating candidates as independent. k = 0 means predict 'no match'."""
    n = len(ps)
    suffix = [None] * (n + 1)
    suffix[n] = np.ones(1)
    for k in range(n - 1, -1, -1):
        nxt = suffix[k + 1]
        cur = np.zeros(len(nxt) + 1)
        cur[:-1] += nxt * (1 - ps[k])
        cur[1:] += nxt * ps[k]
        suffix[k] = cur
    best_k, best_e = 0, suffix[0][0]          # E[F] of empty = P(no true match)
    for k in range(1, min(n, max_k) + 1):
        pre, rest = _pb_pmf(ps[:k]), suffix[k]
        a = np.arange(k + 1)[:, None]
        b = np.arange(len(rest))[None, :]
        f = 1.25 * a / (0.25 * (a + b) + k)
        e = float((pre[:, None] * rest[None, :] * f).sum())
        if e > best_e:
            best_k, best_e = k, e
    return best_k


def select_expected_f(scored, gamma: float = 1.0, min_p: float = 0.01) -> dict:
    out = {}
    for sid, r, p in _groups(scored):
        p = np.clip(p, 0, 1) ** gamma
        keep = p >= min_p
        r, p = r[keep], p[keep]
        if len(p) == 0:
            continue
        k = best_k_expected_f05(p)
        if k:
            out[sid] = list(r[:k])
    return out


def apply(scored: pd.DataFrame, cfg: dict) -> dict:
    if cfg.get("one_owner"):
        scored = one_owner(scored)
    if cfg["method"] == "threshold":
        return select_threshold(scored, cfg["t"], cfg["rel"])
    return select_expected_f(scored, cfg["gamma"])


def tune(scored: pd.DataFrame, truth: dict) -> tuple:
    """Grid-search the post-processing on out-of-fold probabilities with the exact metric."""
    grid = []
    for oo in (False, True):
        for t, rel in itertools.product(np.arange(0.2, 0.96, 0.05), (0.0, 0.5, 0.8, 0.9)):
            grid.append({"method": "threshold", "one_owner": oo, "t": round(float(t), 2), "rel": rel})
        for gamma in (0.8, 1.0, 1.25, 1.5):
            grid.append({"method": "expected_f", "one_owner": oo, "gamma": gamma})
    results = []
    for cfg in grid:
        results.append((macro_f05(apply(scored, cfg), truth), cfg))
    results.sort(key=lambda x: -x[0])
    return results[0], results
