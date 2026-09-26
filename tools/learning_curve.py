"""Learning curve for the stage-1 model: does more training data help, and how much?
Trains on nested samples of S1 records and scores one fixed held-out set of S1 records.

    cd code/business_entity_resolution && python ../../tools/learning_curve.py
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src import decide, pairfeat  # noqa: E402
from src.run import SEED, build_matrix, label  # noqa: E402

cache = Path("../../cache")
norm, cheap = cache / "train_norm.parquet", cache / "train_cheap.parquet"
truth = pl.read_parquet(cache / "train_truth_rid.parquet")
rates = pl.read_parquet(cache / "added_word_rates.parquet")
s1 = pl.read_parquet(norm, columns=["rid", "src"]).filter(pl.col("src") == 1)["rid"].shuffle(seed=SEED + 99)
n = len(s1)
valid_ids = s1[: int(0.03 * n)].sort()
pool = s1[int(0.03 * n): int(0.13 * n)]            # 10% of S1 records, nested subsets taken from its front
t0 = time.time()
Xv, kv, cols = build_matrix(norm, cheap, valid_ids, rates, cache / "lc_valid.npy")
Xp, kp, _ = build_matrix(norm, cheap, pool.sort(), rates, cache / "lc_pool.npy")
yv, yp = label(kv, truth), label(kp, truth)
print(f"features built in {time.time() - t0:.0f}s: valid {len(yv):,} pairs, pool {len(yp):,} pairs", flush=True)
truth_v = truth.filter(pl.col("i").is_in(valid_ids.implode()))
params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, seed=SEED, verbose=-1)
pool_i = kp["i"].to_numpy()
import json, os
configs = json.loads(os.environ.get("LC_CONFIGS", "null")) or [{}]
fracs = json.loads(os.environ.get("LC_FRACS", "[0.25, 0.5, 1.0]"))
for frac, cfg in [(f, c) for c in configs for f in fracs]:
    params = {**dict(objective="binary", learning_rate=0.1, num_leaves=127, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, seed=SEED, verbose=-1), **cfg}
    keep_ids = pool[: int(len(pool) * frac)]
    rows = np.nonzero(np.isin(pool_i, keep_ids.to_numpy()))[0]
    ds = lgb.Dataset(np.asarray(Xp[rows]), yp[rows], params={"max_bin": 127, "verbose": -1})
    m = lgb.train(params, ds, 6000, valid_sets=[lgb.Dataset(np.asarray(Xv), yv)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(np.asarray(Xv), num_iteration=m.best_iteration)
    sc = kv.with_columns(pl.Series("p", p))
    f = decide.macro_f05(decide.apply(sc, {"method": "expected_f", "one_owner": True, "gamma": 1.0, "floor": 0.1}),
                         truth_v, valid_ids)
    ll = -np.mean(yv * np.log(np.clip(p, 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-7, 1)))
    print(f"{cfg} train on {frac * 10:.1f}% of S1 ({len(rows):,} pairs): best_iter={m.best_iteration}  "
          f"valid logloss={ll:.5f}  stage-1 F0.5={f:.4f}", flush=True)
    del ds, m
for f in ("lc_valid.npy", "lc_pool.npy"):
    (cache / f).unlink()
print(f"done in {time.time() - t0:.0f}s")
