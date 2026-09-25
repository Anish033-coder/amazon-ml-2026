"""End-to-end pipeline: data -> normalisation -> blocking -> features -> LightGBM -> decision -> output.

    python -m src.run --data-dir ../../student_resource/dataset --out-dir ../../output

Intermediate results are cached as parquet in --cache-dir, so later runs skip finished
stages (delete the cache directory to recompute everything). Every stage reads only the
columns / rows it needs, so the full data (~12M records per split) fits in 16 GB RAM.
"""
import argparse
import csv
import gc
import json
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from . import block, decide, pairfeat, prep

SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_split(split: str, args, cache: Path):
    """Normalise, block and compute cheap scores for one split; everything is cached."""
    norm_path = cache / f"{split}_norm.parquet"
    if not norm_path.exists():
        prep.prepare_split(args.data_dir, split, cache)
        gc.collect()
    pairs_path = cache / f"{split}_pairs.parquet"
    if not pairs_path.exists():
        cols = ["rid", "src", "country", "core", "name_compact", "addr_nums", "addr_words"]
        block.build_candidates(pl.read_parquet(norm_path, columns=cols), cap_s1=args.cap_s1,
                               cap_r=args.cap_r, max_per_s1=args.max_per_s1).write_parquet(pairs_path)
        gc.collect()
    cheap_path = cache / f"{split}_cheap.parquet"
    if not cheap_path.exists():
        pairfeat.cheap_scores(norm_path, pl.read_parquet(pairs_path)).write_parquet(cheap_path)
        gc.collect()
    log(f"{split}: normalised, blocked, cheap-scored")
    return norm_path, cheap_path


def truth_rid(norm_path, data_dir) -> pl.DataFrame:
    ids = pl.read_parquet(norm_path, columns=["rid", "entity_id"])
    return (prep.load_truth_pairs(data_dir)
                .join(ids.rename({"entity_id": "s1_id", "rid": "i"}), on="s1_id")
                .join(ids.rename({"entity_id": "r_id", "rid": "j"}), on="r_id").select("i", "j"))


def lgb_params(args):
    return dict(objective="binary", learning_rate=args.lr, num_leaves=127, min_child_samples=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                max_bin=127, seed=SEED, deterministic=True, verbose=-1, num_threads=0)


def write_lists(path: Path, s1: pl.DataFrame, pairs: pl.DataFrame, ids: pl.DataFrame, col: str):
    """One row per test S1 record (file order); comma-joined S2/S3 ids; empty when none."""
    lists = (pairs.join(ids.rename({"rid": "j", "entity_id": "rid_j"}), on="j")
                  .group_by("i").agg(pl.col("rid_j").unique().sort().str.join(",").alias(col)))
    out = (s1.select(pl.col("rid").alias("i"), pl.col("entity_id").alias("source1_entity_id"))
             .join(lists, on="i", how="left", maintain_order="left")
             .select("source1_entity_id", pl.col(col).fill_null("")))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="../../cache")
    ap.add_argument("--out-dir", default="../../output")
    ap.add_argument("--sample-frac", type=float, default=0.1, help="share of train S1 records used for the model")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--cap-s1", type=int, default=6)
    ap.add_argument("--cap-r", type=int, default=40)
    ap.add_argument("--max-per-s1", type=int, default=30)
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--log", default="../../experiments/log.csv")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # ---------------- train: full features on a sample of S1 records, grouped CV
    norm_path, cheap_path = build_split("train", args, cache)
    truth = truth_rid(norm_path, args.data_dir)
    meta = pl.read_parquet(norm_path, columns=["rid", "src", "country"]).filter(pl.col("src") == 1)
    rep = block.recall_report(pl.read_parquet(cheap_path, columns=["i", "j"]), truth, len(meta))
    log(f"blocking: {rep}")
    s1 = meta["rid"].sample(fraction=args.sample_frac, seed=SEED).sort()
    sub = pl.scan_parquet(cheap_path).filter(pl.col("i").is_in(s1.implode())).collect()
    rec = pairfeat.record_table(norm_path)
    feats = pairfeat.full_features(rec, sub)
    del rec, sub
    gc.collect()
    y = (feats.select("i", "j").join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["i", "j"],
                                     how="left", maintain_order="left")["y"].fill_null(0).to_numpy())
    cols = pairfeat.feature_columns(feats)
    X = feats.select(cols).to_numpy().astype(np.float32)
    groups = feats["i"].to_numpy()
    scored = feats.select("i", "j")
    del feats
    gc.collect()
    log(f"model data: {len(X):,} pairs from {len(s1):,} S1 records, {y.mean():.3f} positive, {len(cols)} features")

    oof, iters = np.zeros(len(y)), []
    for k, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups)):
        m = lgb.train(lgb_params(args), lgb.Dataset(X[a], y[a]), 3000,
                      valid_sets=[lgb.Dataset(X[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[b] = m.predict(X[b], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"fold {k}: best_iter={m.best_iteration}")
    scored = scored.with_columns(pl.Series("p", oof))
    scored.write_parquet(cache / "train_oof.parquet")
    truth_s = truth.filter(pl.col("i").is_in(s1.implode()))
    best_score, best_cfg = decide.tune(scored, truth_s, s1)
    log(f"OOF macro F0.5 = {best_score:.4f} with {best_cfg}")
    pred = decide.apply(scored, best_cfg)
    for c in meta["country"].unique().sort():
        ids = meta.filter((pl.col("country") == c) & pl.col("rid").is_in(s1.implode()))["rid"]
        log(f"  country={c:<8} F0.5={decide.macro_f05(pred, truth_s, ids):.4f}")

    model = lgb.train(lgb_params(args), lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    model.save_model(str(cache / "model.txt"))
    (cache / "decision.json").write_text(json.dumps({"cfg": best_cfg, "cols": cols}))
    imp = sorted(zip(model.feature_importance("gain"), cols), reverse=True)[:15]
    log("top features: " + ", ".join(c for _, c in imp))
    del X, y, groups, truth, scored
    gc.collect()

    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "note": args.note, **rep,
           "sample_frac": args.sample_frac, "oof_f05": round(best_score, 4),
           "decision": json.dumps(best_cfg), "lb_public": ""}
    try:
        row["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        row["commit"] = ""

    # ---------------- test: score every candidate pair, chunk by chunk
    if not args.skip_test:
        tnorm_path, tcheap_path = build_split("test", args, cache)
        rec = pairfeat.record_table(tnorm_path)
        n = pl.scan_parquet(tcheap_path).select(pl.len()).collect().item()
        ps, step = [], 3_000_000
        for s in range(0, n, step):
            chunk = pl.scan_parquet(tcheap_path).slice(s, step).collect()
            f = pairfeat.full_features(rec, chunk)
            ps.append(f.select("i", "j").with_columns(
                pl.Series("p", model.predict(f.select(cols).to_numpy().astype(np.float32)))))
            del f, chunk
            gc.collect()
            log(f"  scored {min(s + step, n):,}/{n:,} test pairs")
        del rec
        tscored = pl.concat(ps)
        tscored.write_parquet(cache / "test_scored.parquet")
        matches = decide.apply(tscored, best_cfg)
        tmeta = pl.read_parquet(tnorm_path, columns=["rid", "entity_id", "src", "country"])
        ts1 = tmeta.filter(pl.col("src") == 1)
        out = Path(args.out_dir)
        write_lists(out / "matching_results.tsv", ts1, matches, tmeta.select("rid", "entity_id"), "matched_entity_ids")
        write_lists(out / "candidate_pairs.tsv", ts1, tscored.select("i", "j"), tmeta.select("rid", "entity_id"),
                    "candidate_entity_ids")
        per = (ts1.select(pl.col("rid").alias("i"), "country")
                  .join(matches.group_by("i").len("n"), on="i", how="left").fill_null(0))
        for r in per.group_by("country").agg(pl.col("n").mean(), (pl.col("n") == 0).mean().alias("e")).sort("country").iter_rows():
            log(f"  test country={r[0]:<8} avg matches={r[1]:.2f} empty share={r[2]:.3f}")
        row["test_cands_per_s1"] = round(n / len(ts1), 2)
        log(f"wrote {out / 'matching_results.tsv'} and {out / 'candidate_pairs.tsv'}")

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(args.log).exists()
    with open(args.log, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "commit", "note", "pair_recall", "all_found_share", "cands_per_s1",
                                           "sample_frac", "oof_f05", "decision", "test_cands_per_s1", "lb_public"],
                           extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
