"""End-to-end: data -> blocking -> features -> LightGBM (grouped CV) -> post-processing -> output.

    python -m src.pipeline --data-dir <student_resource>/dataset --out-dir <repo>/output
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import postprocess
from .blocking import build_candidates, candidates_by_s1, fit_views
from .features import pair_features
from .io_utils import add_normalized, load_ground_truth, load_sources, write_id_lists
from .metric import blocking_report, macro_f05

SEED = 42
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=SEED, deterministic=True, verbose=-1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(data_dir, split, args):
    df = add_normalized(load_sources(data_dir, split))
    mats = fit_views(df)
    pairs = build_candidates(df, mats, k_fwd=args.k_fwd, k_rev=args.k_rev,
                             by_country=not args.no_country_block)
    feats = pair_features(df, mats, pairs)
    ids = df["entity_id"].to_numpy()
    pairs = pairs.assign(s1=ids[pairs["i"]], r=ids[pairs["j"]],
                         country=df["country_norm"].to_numpy()[pairs["i"]])
    return df, pairs, feats


def train_cv(pairs, feats, y, groups, n_splits):
    oof = np.zeros(len(y))
    iters = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(feats, y, groups)):
        dtr = lgb.Dataset(feats.iloc[tr], y[tr])
        dva = lgb.Dataset(feats.iloc[va], y[va])
        m = lgb.train(LGB_PARAMS, dtr, 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(feats.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"fold {fold}: best_iter={m.best_iteration}")
    return oof, iters


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--k-fwd", type=int, default=10)
    ap.add_argument("--k-rev", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-country-block", action="store_true")
    ap.add_argument("--loco", action="store_true",
                    help="also report leave-one-country-out F0.5 (proxy for unseen France)")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--log", default="experiments/log.csv")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()

    log("train: loading + blocking + features")
    df, pairs, feats = prepare(args.data_dir, "train", args)
    truth = load_ground_truth(args.data_dir)
    s1_train = df.loc[df["source"] == 1, "entity_id"].tolist()
    truth = {i: truth.get(i, set()) for i in s1_train}
    br = blocking_report(candidates_by_s1(df, pairs), truth, int((df["source"] != 1).sum()))
    log(f"blocking: {json.dumps({k: round(v, 4) for k, v in br.items()})}")
    log(f"baseline all-empty F0.5 = {macro_f05({}, truth):.4f}")

    y = np.array([r in truth[s] for s, r in zip(pairs["s1"], pairs["r"])], dtype=np.int8)
    log(f"pairs={len(pairs):,} positives={y.sum():,} features={feats.shape[1]}")

    oof, iters = train_cv(pairs, feats, y, pairs["s1"].to_numpy(), args.folds)
    scored = pairs[["s1", "r"]].assign(p=oof)
    (best_score, best_cfg), results = postprocess.tune(scored, truth)
    log(f"OOF macro F0.5 = {best_score:.4f} with {best_cfg}")
    for method in ("threshold", "expected_f"):
        s, c = next(x for x in results if x[1]["method"] == method)
        log(f"  best {method:<10} {s:.4f}  {c}")
    pred = postprocess.apply(scored, best_cfg)
    countries = df.loc[df["source"] == 1].set_index("entity_id")["country_norm"]
    for c in sorted(countries.unique()):
        ids = countries.index[countries == c]
        log(f"  country={c:<10} n={len(ids):>7,}  F0.5={macro_f05(pred, truth, ids):.4f}")

    if args.loco:
        c_groups = pairs["country"].to_numpy()
        if len(np.unique(c_groups)) > 1:
            loco_oof, _ = train_cv(pairs, feats, y, c_groups, len(np.unique(c_groups)))
            loco_pred = postprocess.apply(pairs[["s1", "r"]].assign(p=loco_oof), best_cfg)
            for c in sorted(np.unique(c_groups)):
                ids = countries.index[countries == c]
                log(f"  LOCO held-out country={c:<10} F0.5={macro_f05(loco_pred, truth, ids):.4f}")

    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "commit": git_commit(), "note": args.note,
           "k_fwd": args.k_fwd, "k_rev": args.k_rev, "country_block": not args.no_country_block,
           **{k: round(v, 4) for k, v in br.items()}, "oof_f05": round(best_score, 4),
           "post": json.dumps(best_cfg), "test_cands_per_s1": "", "lb_public": ""}

    if not args.skip_test:
        n_iter = int(np.mean(iters) * 1.1)
        log(f"final model on all train pairs, {n_iter} rounds")
        model = lgb.train(LGB_PARAMS, lgb.Dataset(feats, y), n_iter)
        log("test: loading + blocking + features")
        tdf, tpairs, tfeats = prepare(args.data_dir, "test", args)
        tscored = tpairs[["s1", "r"]].assign(p=model.predict(tfeats[feats.columns]))
        matches = postprocess.apply(tscored, best_cfg)
        cands = candidates_by_s1(tdf, tpairs)
        s1_test = tdf.loc[tdf["source"] == 1, "entity_id"].tolist()
        out = Path(args.out_dir)
        write_id_lists(out / "matching_results.tsv", s1_test, matches, "matched_entity_ids")
        write_id_lists(out / "candidate_pairs.tsv", s1_test, cands, "candidate_entity_ids")
        tc = tdf.loc[tdf["source"] == 1, "country_norm"]
        n_match = pd.Series({s: len(matches.get(s, [])) for s in s1_test})
        for c in sorted(tc.unique()):
            ids = tdf.loc[(tdf["source"] == 1) & (tdf["country_norm"] == c), "entity_id"]
            m = n_match[ids.to_numpy()]
            log(f"  test country={c:<10} n={len(ids):>7,}  predicted non-empty={np.mean(m > 0):.3f}"
                f"  avg matches={m.mean():.2f}")
        log(f"wrote {out/'matching_results.tsv'} and {out/'candidate_pairs.tsv'}")
        row["test_cands_per_s1"] = round(len(tpairs) / max(len(s1_test), 1), 2)

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(args.log).exists()
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
