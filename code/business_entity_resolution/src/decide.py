"""From pair probabilities to per-Source-1 match lists, vectorised with polars.

The metric is macro F0.5 over S1 records, so every decision is made per S1 group:
  * one_owner  - each S2/S3 record keeps only its best S1 (S1 is deduplicated, and in
                 the training labels no S2/S3 record ever matches two S1 records)
  * threshold  - keep p >= t and p >= rel * best p of the group
  * expected_f - keep the top-k that maximises an estimate of the expected F0.5,
                 E[F(k)] ~ 1.25 * sum_topk(p) / (0.25 * sum_all(p) + k),
                 or nothing when P(no match) = prod(1 - p) is higher
"""
import itertools

import numpy as np
import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.Series) -> float:
    """Exact leaderboard metric. pred/truth: (i, j) pairs; s1: every evaluated S1 rid."""
    tp = pred.join(truth, on=["i", "j"], how="semi").group_by("i").len("tp")
    base = (pl.DataFrame({"i": s1})
              .join(pred.group_by("i").len("np"), on="i", how="left")
              .join(truth.group_by("i").len("nt"), on="i", how="left")
              .join(tp, on="i", how="left").fill_null(0))
    f = (pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
           .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("nt") + pl.col("np"))))
    return float(base.select(f.mean()).item())


def one_owner(scored: pl.DataFrame) -> pl.DataFrame:
    return (scored.sort(["j", "p"], descending=[False, True])
                  .unique("j", keep="first", maintain_order=False))


def threshold(scored: pl.DataFrame, t: float, rel: float) -> pl.DataFrame:
    return scored.filter((pl.col("p") >= t) & (pl.col("p") >= rel * pl.col("p").max().over("i")))


def expected_f(scored: pl.DataFrame, gamma: float = 1.0, floor: float = 0.0) -> pl.DataFrame:
    s = (scored.with_columns((pl.col("p").clip(1e-6, 1 - 1e-6) ** gamma).alias("q"))
               .sort(["i", "q"], descending=[False, True])
               .with_columns(pl.int_range(1, pl.len() + 1).over("i").alias("k"),
                             pl.col("q").cum_sum().over("i").alias("cum"),
                             pl.col("q").sum().over("i").alias("tot"),
                             (1 - pl.col("q")).log().sum().over("i").exp().alias("p0"))
               .with_columns((1.25 * pl.col("cum") / (0.25 * pl.col("tot") + pl.col("k"))).alias("ef")))
    best = s.group_by("i").agg(pl.col("k").get(pl.col("ef").arg_max()).alias("kb"),
                               pl.col("ef").max().alias("efb"), pl.col("p0").first().alias("p0b"))
    return (s.join(best, on="i")
             .filter((pl.col("k") <= pl.col("kb")) & (pl.col("efb") > pl.col("p0b") + floor))
             .select(scored.columns))


def apply(scored: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    s = one_owner(scored) if cfg.get("one_owner") else scored
    if cfg["method"] == "threshold":
        return threshold(s, cfg["t"], cfg["rel"]).select("i", "j")
    return expected_f(s, cfg["gamma"], cfg.get("floor", 0.0)).select("i", "j")


def tune(scored: pl.DataFrame, truth: pl.DataFrame, s1: pl.Series, verbose=True):
    """Grid search with the exact metric. `scored` may hold every pair of the split (so the
    one-owner rule sees all competitors); only the S1 records in `s1` are evaluated."""
    res = []
    keep = pl.col("i").is_in(s1.implode())
    for oo in (True, False):
        base = (one_owner(scored) if oo else scored).filter(keep)
        grid = [{"method": "expected_f", "one_owner": oo, "gamma": g, "floor": f}
                for g, f in itertools.product((0.7, 1.0, 1.4, 2.0), (0.0, 0.1, 0.2))]
        grid += [{"method": "threshold", "one_owner": oo, "t": float(t), "rel": rel}
                 for t, rel in itertools.product(np.round(np.arange(0.3, 0.91, 0.1), 2), (0.0, 0.5))]
        for cfg in grid:
            sel = (threshold(base, cfg["t"], cfg["rel"]) if cfg["method"] == "threshold"
                   else expected_f(base, cfg["gamma"], cfg["floor"]))
            res.append((macro_f05(sel.select("i", "j"), truth, s1), cfg))
    res.sort(key=lambda x: -x[0])
    if verbose:
        for m in ("expected_f", "threshold"):
            for oo in (True, False):
                sc, c = next(x for x in res if x[1]["method"] == m and x[1]["one_owner"] == oo)
                print(f"    {m:<10} one_owner={oo!s:<5} F0.5={sc:.4f} {c}", flush=True)
    return res[0]
