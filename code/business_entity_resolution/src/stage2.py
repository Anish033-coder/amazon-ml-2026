"""Second-stage features: how a pair's first-stage probability p1 compares with its
competitors, and whether the S2/S3 record agrees with the S1 record's other likely matches.

Within an S1 group (all candidates of one S1 record):
  rank, gap to the best, share of the group's mass, how many strong candidates there are.
Within an S2/S3 group (all S1 records competing for one S2/S3 record):
  rank, best competing p1 from another S1 record, number of strong competitors.
Sibling agreement: true S2/S3 copies of one business resemble each other, so each
candidate is compared (name and address) with the S1 record's most likely other match.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

STAGE2_CHEAP = ["n_keys", "c_name", "c_addr", "c_combo", "n_cand_i", "n_cand_j"]


def group_features(scored: pl.DataFrame) -> pl.DataFrame:
    p = pl.col("p1")
    top2 = (scored.group_by("j").agg(p.max().alias("_m1"),
                                     p.sort(descending=True).get(1, null_on_oob=True).alias("_m2")))
    out = scored.join(top2, on="j", how="left").with_columns(
        p.rank("min", descending=True).over("i").cast(pl.Float32).alias("p1_rank_i"),
        (p.max().over("i") - p).alias("p1_gap_i"),
        (p / p.sum().over("i").clip(1e-6)).alias("p1_share_i"),
        p.sum().over("i").alias("p1_sum_i"),
        (p > 0.5).sum().over("i").cast(pl.Float32).alias("p1_n50_i"),
        (p > 0.1).sum().over("i").cast(pl.Float32).alias("p1_n10_i"),
        p.rank("min", descending=True).over("j").cast(pl.Float32).alias("p1_rank_j"),
        (p > 0.5).sum().over("j").cast(pl.Float32).alias("p1_n50_j"))
    # best p1 among *other* S1 records competing for the same S2/S3 record
    out = out.with_columns(
        pl.when(p < pl.col("_m1")).then(pl.col("_m1")).otherwise(pl.col("_m2").fill_null(0.0))
          .cast(pl.Float32).alias("p1_best_other_j"))
    return out.with_columns((p - pl.col("p1_best_other_j")).alias("p1_margin_j")).drop("_m1", "_m2")


def sibling_features(scored: pl.DataFrame, norm_path, chunk: int = 5_000_000) -> pl.DataFrame:
    """Similarity of each candidate to the S1 record's best *other* candidate (by p1)."""
    top = (scored.sort(["i", "p1"], descending=[False, True])
                 .group_by("i", maintain_order=True).head(2)
                 .with_columns(pl.int_range(pl.len()).over("i").alias("k"))
                 .pivot(on="k", index="i", values="j"))
    top = top.rename({c: f"top{c}" for c in top.columns if c != "i"})
    if "top1" not in top.columns:
        top = top.with_columns(pl.lit(None, pl.UInt32).alias("top1"))
    s = scored.join(top, on="i", how="left").with_columns(
        pl.when(pl.col("top0") == pl.col("j")).then(pl.col("top1")).otherwise(pl.col("top0")).alias("sib"))
    t = pl.read_parquet(norm_path, columns=["rid", "name_core", "addr_norm"]).sort("rid")
    name, addr = t["name_core"], t["addr_norm"]
    del t
    sn, sa = [], []
    for a in range(0, len(s), chunk):
        c = s.slice(a, chunk)
        j = c["j"].to_numpy()
        sib = c["sib"].fill_null(c["j"]).to_numpy()   # no sibling -> compare with itself, flagged below
        sn.append(process.cpdist(name.gather(j).to_list(), name.gather(sib).to_list(),
                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32))
        sa.append(process.cpdist(addr.gather(j).to_list(), addr.gather(sib).to_list(),
                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32))
    has = s["sib"].is_not_null()
    return s.select("i", "j").with_columns(
        pl.Series("sib_name", np.concatenate(sn)).cast(pl.Float32),
        pl.Series("sib_addr", np.concatenate(sa)).cast(pl.Float32)).with_columns(
        pl.when(has).then(pl.col("sib_name")).otherwise(None).alias("sib_name"),
        pl.when(has).then(pl.col("sib_addr")).otherwise(None).alias("sib_addr"))


def build(scored: pl.DataFrame, cheap_path, norm_path) -> pl.DataFrame:
    """scored: (i, j, p1) for every candidate pair of a split. Returns stage-2 feature frame."""
    g = group_features(scored)
    sib = sibling_features(scored, norm_path)
    cheap = pl.read_parquet(cheap_path, columns=["i", "j"] + STAGE2_CHEAP)
    return g.join(sib, on=["i", "j"], how="left").join(cheap, on=["i", "j"], how="left")


def columns(frame: pl.DataFrame) -> list:
    return [c for c in frame.columns if c not in ("i", "j")]
