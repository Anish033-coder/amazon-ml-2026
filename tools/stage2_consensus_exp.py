"""Experiment: do 'consensus' features help the stage-2 model?

For each S1 record, the copies the stage-1 model is most sure about define a consensus
house number and name. A sister business sits a few doors from that consensus; a true copy
agrees with it. Stage 2 is retrained on v10's saved stage-1 scores with and without these
features, on the same S1 sample v10 was evaluated on, and the out-of-fold F0.5 compared.

    cd code/business_entity_resolution && python ../../tools/stage2_consensus_exp.py
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src import decide, stage2  # noqa: E402
from src.run import SEED, label  # noqa: E402

cache = Path("../../cache")
norm, cheap = cache / "train_norm.parquet", cache / "train_cheap.parquet"
t0 = time.time()
truth = pl.read_parquet(cache / "train_truth_rid.parquet")
meta = pl.read_parquet(norm, columns=["rid", "src"]).filter(pl.col("src") == 1)
s1b = meta["rid"].sample(fraction=0.35, seed=SEED + 1).sort()      # v10's stage-2 sample
p1 = pl.read_parquet(cache / "train_scored_v10.parquet", columns=["i", "j", "p1"])
f2_path = cache / "exp_f2.parquet"
stage2.build(p1, cheap, norm, f2_path)
del p1
fb = pl.scan_parquet(f2_path).filter(pl.col("i").is_in(s1b.implode())).collect()
f2_path.unlink()
print(f"stage-2 base features for sample B: {fb.shape} in {time.time() - t0:.0f}s", flush=True)

# ---- consensus features (per S1 group, from the copies stage 1 is most sure about)
rec = pl.read_parquet(norm, columns=["rid", "compact_key", "addr_first_num", "addr_norm"])
g = fb.select("i", "j", "p1").join(rec.rename({"rid": "j"}), on="j", how="left")
strong = g.filter(pl.col("p1") > 0.5)
num_votes = (strong.filter(pl.col("addr_first_num") != "").group_by("i", "addr_first_num").agg(pl.col("p1").sum().alias("w"))
                   .sort(["i", "w"], descending=[False, True]).group_by("i", maintain_order=True).first()
                   .select("i", pl.col("addr_first_num").alias("cons_num")))
name_votes = (strong.group_by("i", "compact_key").agg(pl.col("p1").sum().alias("w"))
                    .sort(["i", "w"], descending=[False, True]).group_by("i", maintain_order=True).first()
                    .select("i", pl.col("compact_key").alias("cons_name")))
g = g.join(num_votes, on="i", how="left").join(name_votes, on="i", how="left")
ni = pl.col("addr_first_num").str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
nc = pl.col("cons_num").str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
g = g.with_columns(
    pl.when((pl.col("addr_first_num") == "") | pl.col("cons_num").is_null()).then(None)
      .otherwise((pl.col("addr_first_num") == pl.col("cons_num")).cast(pl.Float32)).alias("cons_num_eq"),
    (ni - nc).abs().cast(pl.Float32).log1p().alias("cons_num_offset_log"),
    pl.when(pl.col("cons_name").is_null()).then(None)
      .otherwise((pl.col("compact_key") == pl.col("cons_name")).cast(pl.Float32)).alias("cons_name_eq"),
    # how many OTHER strong copies share this copy's exact name / house number / address
    ((pl.col("p1") > 0.5).cast(pl.Int32).sum().over(["i", "compact_key"]) - (pl.col("p1") > 0.5).cast(pl.Int32)).cast(pl.Float32).alias("cons_same_name_n"),
    ((pl.col("p1") > 0.5).cast(pl.Int32).sum().over(["i", "addr_first_num"]) - (pl.col("p1") > 0.5).cast(pl.Int32)).cast(pl.Float32).alias("cons_same_num_n"),
    (pl.when(pl.col("addr_norm") != "").then((pl.col("p1") > 0.5).cast(pl.Int32).sum().over(["i", "addr_norm"]) - (pl.col("p1") > 0.5).cast(pl.Int32)))
      .cast(pl.Float32).alias("cons_same_addr_n"))
cons_cols = ["cons_num_eq", "cons_num_offset_log", "cons_name_eq", "cons_same_name_n", "cons_same_num_n", "cons_same_addr_n"]
fb = fb.join(g.select(["i", "j"] + cons_cols), on=["i", "j"], how="left")
y = label(fb, truth)
groups = fb["i"].to_numpy()
base_cols = [c for c in fb.columns if c not in ("i", "j") and c not in cons_cols]
truth_b = truth.filter(pl.col("i").is_in(s1b.implode()))
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, seed=SEED, verbose=-1)
print(f"features ready in {time.time() - t0:.0f}s; {len(y):,} pairs; base {len(base_cols)} + consensus {len(cons_cols)}", flush=True)
for tag, cols in (("v10 stage 2 (baseline)", base_cols), ("+ consensus features", base_cols + cons_cols)):
    X = fb.select(cols).to_numpy().astype(np.float32)
    oof = np.zeros(len(y))
    for a, b in GroupKFold(n_splits=3).split(X, y, groups):
        m = lgb.train(params, lgb.Dataset(X[a], y[a]), 3000, valid_sets=[lgb.Dataset(X[b], y[b])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[b] = m.predict(X[b], num_iteration=m.best_iteration)
    scored = fb.select("i", "j").with_columns(pl.Series("p", oof))
    best = max((decide.macro_f05(decide.apply(scored, {"method": "expected_f", "one_owner": True, "gamma": g_, "floor": f_}),
                                 truth_b, s1b), g_, f_) for g_ in (1.0, 1.4) for f_ in (0.1, 0.2))
    print(f"  {tag:<26} OOF F0.5 = {best[0]:.4f} (gamma={best[1]}, floor={best[2]})  [{time.time() - t0:.0f}s]", flush=True)
    del X
