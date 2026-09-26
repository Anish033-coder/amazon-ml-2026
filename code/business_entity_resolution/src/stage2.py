"""Second-stage features: how a pair's first-stage probability p1 compares with its
competitors, and whether the S2/S3 record agrees with the S1 record's other likely matches.

Within an S1 group (all candidates of one S1 record):
  rank, gap to the best, share of the group's mass, how many strong candidates there are.
Within an S2/S3 group (all S1 records competing for one S2/S3 record):
  rank, best competing p1 from another S1 record, number of strong competitors.
Sibling agreement: true S2/S3 copies of one business resemble each other, so each
candidate is compared (name and address) with the S1 record's most likely other match.
"""
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

STAGE2_CHEAP = ["n_keys", "c_name", "c_addr", "c_combo", "n_cand_i", "n_cand_j"]


def j_stats(scored: pl.DataFrame) -> pl.DataFrame:
    """Per S2/S3 record: its best and second-best p1 and how many S1 records claim it strongly."""
    p = pl.col("p1")
    return scored.group_by("j").agg(p.max().alias("_m1"), p.sort(descending=True).get(1, null_on_oob=True).alias("_m2"),
                                    (p > 0.5).sum().cast(pl.Float32).alias("p1_n50_j"))


def group_features(chunk: pl.DataFrame, js: pl.DataFrame) -> pl.DataFrame:
    """chunk holds complete S1 groups; js = j_stats() over all pairs of the split."""
    p = pl.col("p1")
    out = chunk.join(js, on="j", how="left").with_columns(
        p.rank("min", descending=True).over("i").cast(pl.Float32).alias("p1_rank_i"),
        (p.max().over("i") - p).alias("p1_gap_i"),
        (p / p.sum().over("i").clip(1e-6)).alias("p1_share_i"),
        p.sum().over("i").alias("p1_sum_i"),
        (p > 0.5).sum().over("i").cast(pl.Float32).alias("p1_n50_i"),
        (p > 0.1).sum().over("i").cast(pl.Float32).alias("p1_n10_i"),
        # rank within the S2/S3 record's competitors, clipped at 3 (1 = best, 2 = runner-up)
        pl.when(p >= pl.col("_m1")).then(1.0).when(p >= pl.col("_m2").fill_null(-1.0)).then(2.0).otherwise(3.0)
          .cast(pl.Float32).alias("p1_rank_j"))
    # best p1 among *other* S1 records competing for the same S2/S3 record
    out = out.with_columns(
        pl.when(p < pl.col("_m1")).then(pl.col("_m1")).otherwise(pl.col("_m2").fill_null(0.0))
          .cast(pl.Float32).alias("p1_best_other_j"))
    return out.with_columns((p - pl.col("p1_best_other_j")).alias("p1_margin_j")).drop("_m1", "_m2")


def sibling_features(scored: pl.DataFrame, name: pl.Series, addr: pl.Series, chunk: int = 5_000_000) -> pl.DataFrame:
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


def consensus_features(ch: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    """For each S1 group, the copies stage 1 is most sure about (p1 > 0.5) define a consensus
    house number and name; a sister business sits a few doors from that consensus."""
    g = ch.select("i", "j", "p1").join(rec, on="j", how="left")
    strong = g.filter(pl.col("p1") > 0.5)
    num = (strong.filter(pl.col("addr_first_num") != "").group_by("i", "addr_first_num").agg(pl.col("p1").sum().alias("w"))
                 .sort(["i", "w"], descending=[False, True]).group_by("i", maintain_order=True).first()
                 .select("i", pl.col("addr_first_num").alias("cons_num")))
    name = (strong.group_by("i", "compact_key").agg(pl.col("p1").sum().alias("w"))
                  .sort(["i", "w"], descending=[False, True]).group_by("i", maintain_order=True).first()
                  .select("i", pl.col("compact_key").alias("cons_name")))
    g = g.join(num, on="i", how="left").join(name, on="i", how="left")
    ni = pl.col("addr_first_num").str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
    nc = pl.col("cons_num").str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
    strong_i = (pl.col("p1") > 0.5).cast(pl.Int32)
    return g.select(
        "i", "j",
        pl.when((pl.col("addr_first_num") == "") | pl.col("cons_num").is_null()).then(None)
          .otherwise((pl.col("addr_first_num") == pl.col("cons_num")).cast(pl.Float32)).alias("cons_num_eq"),
        (ni - nc).abs().cast(pl.Float32).log1p().alias("cons_num_offset_log"),
        pl.when(pl.col("cons_name").is_null()).then(None)
          .otherwise((pl.col("compact_key") == pl.col("cons_name")).cast(pl.Float32)).alias("cons_name_eq"),
        (strong_i.sum().over(["i", "compact_key"]) - strong_i).cast(pl.Float32).alias("cons_same_name_n"),
        (strong_i.sum().over(["i", "addr_first_num"]) - strong_i).cast(pl.Float32).alias("cons_same_num_n"),
        pl.when(pl.col("addr_norm") != "").then(strong_i.sum().over(["i", "addr_norm"]) - strong_i)
          .cast(pl.Float32).alias("cons_same_addr_n"))


def build(scored: pl.DataFrame, cheap_path, norm_path, out_path, n_chunks: int = 8):
    """scored: (i, j, p1) for every candidate pair of a split. Writes the stage-2 feature table to
    out_path, processing complete S1 groups in n_chunks slices so memory stays bounded."""
    js = j_stats(scored)
    t = pl.read_parquet(norm_path, columns=["rid", "name_core", "addr_norm", "compact_key", "addr_first_num"]).sort("rid")
    name, addr = t["name_core"], t["addr_norm"]
    rec = t.select(pl.col("rid").alias("j"), "compact_key", "addr_first_num", "addr_norm")
    del t
    edges = np.linspace(0, int(scored["i"].max()) + 1, n_chunks + 1).astype(np.int64)
    parts = []
    for k in range(n_chunks):
        ch = scored.filter((pl.col("i") >= edges[k]) & (pl.col("i") < edges[k + 1]))
        if len(ch) == 0:
            continue
        g = group_features(ch, js).join(consensus_features(ch, rec), on=["i", "j"], how="left")
        sib = sibling_features(ch, name, addr)
        cheap = (pl.scan_parquet(cheap_path).select(["i", "j"] + STAGE2_CHEAP)
                   .filter((pl.col("i") >= edges[k]) & (pl.col("i") < edges[k + 1])).collect())
        part = Path(str(out_path) + f".part{k}")
        g.join(sib, on=["i", "j"], how="left").join(cheap, on=["i", "j"], how="left").write_parquet(part)
        parts.append(part)
        del ch, g, sib, cheap
    pl.concat([pl.scan_parquet(x) for x in parts]).sink_parquet(out_path)
    for x in parts:
        x.unlink()


def columns(frame: pl.DataFrame) -> list:
    return [c for c in frame.columns if c not in ("i", "j")]
