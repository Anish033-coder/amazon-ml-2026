"""Does self-training help a country the model has no labels for? (proxy for France)

Train the stage-1 model on US labels only, treat India as unlabelled, and measure India's
true F0.5. Then add India pairs the model is confident about as pseudo-labels, retrain and
measure again. An India-trained model gives the in-domain upper bound.

    cd code/business_entity_resolution && python ../../tools/selftrain_sim.py
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
from src import decide  # noqa: E402
from src.run import SEED, build_matrix, label  # noqa: E402

cache = Path("../../cache")
norm, cheap = cache / "train_norm.parquet", cache / "train_cheap.parquet"
truth = pl.read_parquet(cache / "train_truth_rid.parquet")
rates = pl.read_parquet(cache / "added_word_rates.parquet")
meta = pl.read_parquet(norm, columns=["rid", "src", "country"]).filter(pl.col("src") == 1)
us = meta.filter(pl.col("country") == "us")["rid"].shuffle(seed=SEED)
ind = meta.filter(pl.col("country") == "india")["rid"].shuffle(seed=SEED)
us_train = us[: int(0.15 * len(us))].sort()
in_pool = ind[: int(0.15 * len(ind))].sort()               # unlabelled target data used for pseudo-labels
in_eval = ind[int(0.15 * len(ind)): int(0.25 * len(ind))].sort()   # held-out India records for scoring
t0 = time.time()
Xu, ku, cols = build_matrix(norm, cheap, us_train, rates, cache / "st_us.npy")
Xp, kp, _ = build_matrix(norm, cheap, in_pool, rates, cache / "st_inpool.npy")
Xe, ke, _ = build_matrix(norm, cheap, in_eval, rates, cache / "st_ineval.npy")
yu, yp, ye = label(ku, truth), label(kp, truth), label(ke, truth)
print(f"features in {time.time() - t0:.0f}s: US {len(yu):,}  India pool {len(yp):,}  India eval {len(ye):,}", flush=True)
params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_child_samples=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, seed=SEED, verbose=-1)
CFG = {"method": "expected_f", "one_owner": True, "gamma": 1.0, "floor": 0.1}
truth_e = truth.filter(pl.col("i").is_in(in_eval.implode()))


def fit(X, y, rounds=600):
    return lgb.train(params, lgb.Dataset(X, y, params={"max_bin": 127, "verbose": -1}), rounds)


def india_f(model, tag):
    p = model.predict(np.asarray(Xe))
    f = decide.macro_f05(decide.apply(ke.with_columns(pl.Series("p", p)), CFG), truth_e, in_eval)
    print(f"  {tag:<52} India F0.5 = {f:.4f}", flush=True)
    return f


m_us = fit(np.asarray(Xu), yu)
base = india_f(m_us, "trained on US labels only")
m_in = fit(np.asarray(Xp), yp)
india_f(m_in, "trained on India labels (in-domain upper bound)")
pp = m_us.predict(np.asarray(Xp))
for hi, lo in ((0.95, 0.05), (0.9, 0.1), (0.98, 0.02)):
    keep = (pp >= hi) | (pp <= lo)
    pseudo = (pp[keep] >= hi).astype(np.int8)
    acc = (pseudo == yp[keep]).mean()
    X = np.vstack([np.asarray(Xu), np.asarray(Xp)[keep]])
    y = np.concatenate([yu, pseudo])
    m = fit(X, y)
    india_f(m, f"US labels + India pseudo-labels (p>={hi} / p<={lo}; {keep.mean():.0%} kept, {acc:.4f} correct)")
    del X, y, m
for f in ("st_us.npy", "st_inpool.npy", "st_ineval.npy"):
    (cache / f).unlink()
print(f"done in {time.time() - t0:.0f}s")
