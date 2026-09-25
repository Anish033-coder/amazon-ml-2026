"""Blocking experiment on the full training split: recall at several per-S1 caps.

    cd code/business_entity_resolution
    python ../../tools/block_eval.py --cap-s1 6 --cap-r 40 --k-num 2
"""
import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src import block  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache-dir", default="../../cache")
ap.add_argument("--cap-s1", type=int, default=6)
ap.add_argument("--cap-r", type=int, default=40)
ap.add_argument("--k-name", type=int, default=4)
ap.add_argument("--k-addr", type=int, default=3)
ap.add_argument("--k-num", type=int, default=2)
ap.add_argument("--save", default="")
a = ap.parse_args()
c = Path(a.cache_dir)
df = pl.read_parquet(c / "train_norm.parquet",
                     columns=["rid", "src", "country", "core", "name_compact", "addr_nums", "addr_words"])
truth = pl.read_parquet(c / "train_truth_rid.parquet")
n_s1 = int((df["src"] == 1).sum())
t = time.time()
pairs = block.build_candidates(df, cap_s1=a.cap_s1, cap_r=a.cap_r, max_per_s1=200,
                               k_name=a.k_name, k_addr=a.k_addr, k_num=a.k_num)
print(f"blocking {time.time() - t:.0f}s  {vars(a)}")
ranked = pairs.with_columns(pl.int_range(1, pl.len() + 1).over("i").alias("rank"))
for cap in (20, 30, 40, 60, 100, 200):
    print(f"  max_per_s1={cap:<4}", block.recall_report(ranked.filter(pl.col("rank") <= cap), truth, n_s1))
if a.save:
    pairs.write_parquet(a.save)
