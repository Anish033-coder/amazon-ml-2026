"""Blocking experiment on the full training split: keys -> cheap-score top-K cut -> F0.5 ceiling.

    cd code/business_entity_resolution
    python ../../tools/block_eval2.py --cap 40
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src import block, pairfeat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache-dir", default="../../cache")
ap.add_argument("--cap", type=int, default=40)
ap.add_argument("--save", default="")
a = ap.parse_args()
c = Path(a.cache_dir)
norm = c / "train_norm.parquet"
t = time.time()
pairs = block.build_candidates(pl.read_parquet(norm, columns=["rid", "src", "country", "core", "name_compact",
                                                              "addr_nums", "addr_words"]), max_per_s1=200)
print(f"blocking {time.time() - t:.0f}s, {len(pairs):,} pairs before the cut", flush=True)
if a.save:
    pairs.write_parquet(a.save)
truth = pl.read_parquet(c / "train_truth_rid.parquet")
s1 = pl.read_parquet(norm, columns=["rid", "src"]).filter(pl.col("src") == 1)["rid"]


def report(sel, tag):
    hit = truth.join(sel.select("i", "j"), on=["i", "j"], how="semi")
    per = (pl.DataFrame({"i": s1}).join(truth.group_by("i").len("nt"), on="i", how="left")
             .join(hit.group_by("i").len("h"), on="i", how="left").fill_null(0))
    ceil = per.select(pl.when(pl.col("nt") == 0).then(1.0)
                        .otherwise(1.25 * pl.col("h") / (0.25 * pl.col("nt") + pl.col("h")).clip(1e-9))).mean().item()
    print(f"  {tag:<24} pair_recall={len(hit) / len(truth):.4f}  f05_ceiling={ceil:.4f}  cands/S1={len(sel) / len(s1):.2f}", flush=True)


report(pairs, "all blocked pairs")
cheap = pairfeat.cheap_scores(norm, pairs, cap=a.cap)
report(cheap, f"after top-{a.cap} cut")
