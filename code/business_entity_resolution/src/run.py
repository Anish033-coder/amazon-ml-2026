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

from . import block, decide, pairfeat, prep, stage2

SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_split(split: str, args, cache: Path):
    """Normalise, block and compute cheap scores for one split; everything is cached.

    With --drop-s1 q (train only) a random share q of S1 records is removed before blocking.
    Their S2/S3 copies stay, now without an owner, which reproduces the test set's larger
    pool of S2/S3 records per S1 record (5.75 vs 4.67 in train)."""
    norm_path = cache / f"{split}_norm.parquet"
    if not norm_path.exists():
        prep.prepare_split(args.data_dir, split, cache)
        gc.collect()
    if split == "train" and args.drop_s1 > 0:
        full, split = norm_path, f"train_drop{int(round(args.drop_s1 * 100))}"
        norm_path = cache / f"{split}_norm.parquet"
        if not norm_path.exists():
            df = pl.read_parquet(full)
            s1 = df.filter(pl.col("src") == 1)["rid"]
            gone = s1.sample(fraction=args.drop_s1, seed=SEED + 7)
            (df.filter(~pl.col("rid").is_in(gone.implode())).drop("rid").with_row_index("rid")
               .write_parquet(norm_path))
            del df
            gc.collect()
    pairs_path = cache / f"{split}_pairs.parquet"
    if not pairs_path.exists():
        cols = ["rid", "src", "country", "core", "name_compact", "addr_nums", "addr_words"]
        block.build_candidates(pl.read_parquet(norm_path, columns=cols), cap_s1=args.cap_s1,
                               cap_r=args.cap_r, max_per_s1=200).write_parquet(pairs_path)
        gc.collect()
    cheap_path = cache / f"{split}_cheap.parquet"
    if not cheap_path.exists():
        pairfeat.cheap_scores(norm_path, pl.read_parquet(pairs_path), cap=args.max_per_s1).write_parquet(cheap_path)
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


def build_matrix(norm_path, cheap_path, s1_ids, rates, path: Path, step=2_000_000):
    """Stage-1 features for every candidate pair of the given S1 records, written chunk by chunk
    into a float32 matrix on disk (memory-mapped), so RAM only ever holds one chunk."""
    sub = pl.scan_parquet(cheap_path).filter(pl.col("i").is_in(s1_ids.implode())).collect()
    rec, tf = pairfeat.record_table(norm_path), pairfeat.token_freqs(norm_path)
    X, cols = None, None
    for s in range(0, len(sub), step):
        f = pairfeat.full_features(rec, sub.slice(s, step), tf, rates, chunk=step)
        if X is None:
            cols = pairfeat.feature_columns(f)
            X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(sub), len(cols)))
        X[s:s + len(f)] = f.select(cols).to_numpy().astype(np.float32)
        del f
        gc.collect()
    X.flush()
    keys = sub.select("i", "j")
    del sub, rec, tf
    gc.collect()
    return np.load(path, mmap_mode="r"), keys, cols


def predict_rows(model, X, rows=None, step=1_000_000, num_iteration=None) -> np.ndarray:
    idx = np.arange(X.shape[0]) if rows is None else rows
    return np.concatenate([model.predict(np.asarray(X[idx[s:s + step]]), num_iteration=num_iteration)
                           for s in range(0, len(idx), step)])


def fit_cv_mm(X, y, groups, args, tag):
    """Grouped K-fold on a disk-backed matrix: one binned LightGBM dataset, folds as subsets."""
    params = lgb_params(args)
    full = lgb.Dataset(X, label=y, params={"max_bin": params["max_bin"], "verbose": -1},
                       free_raw_data=True).construct()   # True: subsets must not copy raw rows into RAM
    oof, iters = np.zeros(len(y)), []
    for k, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(np.zeros(len(y)), y, groups)):
        m = lgb.train(params, full.subset(a), 3000, valid_sets=[full.subset(b)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[b] = predict_rows(m, X, b, num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"{tag} fold {k}: best_iter={m.best_iteration}")
        del m
        gc.collect()
    model = lgb.train(params, full, int(np.mean(iters) * 1.1))
    del full
    gc.collect()
    return oof, model


def fit_cv(X, y, groups, args, tag):
    """Grouped K-fold (by S1 record): out-of-fold predictions plus a model fit on everything."""
    oof, iters = np.zeros(len(y)), []
    for k, (a, b) in enumerate(GroupKFold(n_splits=args.folds).split(X, y, groups)):
        m = lgb.train(lgb_params(args), lgb.Dataset(X[a], y[a]), 3000,
                      valid_sets=[lgb.Dataset(X[b], y[b])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[b] = m.predict(X[b], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"{tag} fold {k}: best_iter={m.best_iteration}")
    model = lgb.train(lgb_params(args), lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    return oof, model


def score_pairs(norm_path, cheap_path, model, cols, rates=None, step=3_000_000) -> pl.DataFrame:
    """Stage-1 probability p1 for every candidate pair of a split, chunk by chunk."""
    rec, tf = pairfeat.record_table(norm_path), pairfeat.token_freqs(norm_path)
    n = pl.scan_parquet(cheap_path).select(pl.len()).collect().item()
    out = []
    for s in range(0, n, step):
        f = pairfeat.full_features(rec, pl.scan_parquet(cheap_path).slice(s, step).collect(), tf, rates)
        out.append(f.select("i", "j").with_columns(
            pl.Series("p1", model.predict(f.select(cols).to_numpy().astype(np.float32))).cast(pl.Float32)))
        del f
        gc.collect()
    log(f"  stage-1 scored {n:,} pairs")
    return pl.concat(out)


def predict_parquet(model, path: Path, cols, step=4_000_000) -> pl.DataFrame:
    """(i, j, p1, p) for every row of a stage-2 feature parquet, read chunk by chunk."""
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    out = []
    for s in range(0, n, step):
        c = pl.scan_parquet(path).slice(s, step).collect()
        out.append(c.select("i", "j", "p1").with_columns(
            pl.Series("p", model.predict(c.select(cols).to_numpy().astype(np.float32))).cast(pl.Float32)))
        del c
        gc.collect()
    return pl.concat(out)


def predict_frame(model, frame: pl.DataFrame, cols, step=5_000_000) -> np.ndarray:
    return np.concatenate([model.predict(frame.slice(s, step).select(cols).to_numpy().astype(np.float32))
                           for s in range(0, len(frame), step)])


def with_oof(all_pairs: pl.DataFrame, sample: pl.DataFrame, col: str) -> pl.DataFrame:
    """Replace in-sample predictions of the training sample by their out-of-fold values."""
    return (all_pairs.join(sample.rename({col: "_oof"}), on=["i", "j"], how="left")
                     .with_columns(pl.coalesce("_oof", col).alias(col)).drop("_oof"))


def label(frame: pl.DataFrame, truth: pl.DataFrame) -> np.ndarray:
    return (frame.select("i", "j").join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["i", "j"],
                                        how="left", maintain_order="left")["y"].fill_null(0).to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="../../cache")
    ap.add_argument("--out-dir", default="../../output")
    ap.add_argument("--frac-a", type=float, default=0.35, help="share of train S1 records for the stage-1 model")
    ap.add_argument("--frac-b", type=float, default=0.35, help="share of train S1 records for the stage-2 model")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--cap-s1", type=int, default=6)
    ap.add_argument("--cap-r", type=int, default=40)
    ap.add_argument("--max-per-s1", type=int, default=40)
    ap.add_argument("--drop-s1", type=float, default=0.0,
                    help="hide this share of train S1 records (test-like conditions)")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--log", default="../../experiments/log.csv")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    t0 = time.time()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # ---------------- train, stage 1: pair model on sample A of S1 records
    norm_path, cheap_path = build_split("train", args, cache)
    truth = truth_rid(norm_path, args.data_dir)
    meta = pl.read_parquet(norm_path, columns=["rid", "src", "country"]).filter(pl.col("src") == 1)
    rep = block.recall_report(pl.read_parquet(cheap_path, columns=["i", "j"]), truth, len(meta))
    log(f"blocking: {rep}")
    s1_a = meta["rid"].sample(fraction=args.frac_a, seed=SEED).sort()
    s1_b = meta["rid"].sample(fraction=args.frac_b, seed=SEED + 1).sort()
    # added-word match rates are learned on S1 records outside both model samples
    s1_c = meta.filter(~pl.col("rid").is_in(s1_a.implode()) & ~pl.col("rid").is_in(s1_b.implode()))["rid"]
    rates = pairfeat.learn_added_word_rates(norm_path, cheap_path, truth, s1_c)
    rates.write_parquet(cache / "added_word_rates.parquet")
    log(f"added-word rates: {len(rates):,} words from {len(s1_c):,} held-out S1 records")
    X, keys_a, cols = build_matrix(norm_path, cheap_path, s1_a, rates, cache / "train_X1.npy")
    y = label(keys_a, truth)
    log(f"stage 1: {len(y):,} pairs from {len(s1_a):,} S1 records, {y.mean():.3f} positive, {len(cols)} features")
    oof1, model1 = fit_cv_mm(X, y, keys_a["i"].to_numpy(), args, "stage 1")
    model1.save_model(str(cache / "model1.txt"))
    del X, y
    gc.collect()
    (cache / "train_X1.npy").unlink()
    p1 = with_oof(score_pairs(norm_path, cheap_path, model1, cols, rates), keys_a.with_columns(pl.Series("p1", oof1).cast(pl.Float32)), "p1")

    # ---------------- train, stage 2: competition + sibling features on sample B
    f2_path = cache / "train_f2.parquet"
    stage2.build(p1, cheap_path, norm_path).write_parquet(f2_path)
    del p1
    gc.collect()
    cols2 = [c for c in pl.read_parquet_schema(f2_path) if c not in ("i", "j")]
    fb = pl.scan_parquet(f2_path).filter(pl.col("i").is_in(s1_b.implode())).collect()
    X2, y2 = fb.select(cols2).to_numpy().astype(np.float32), label(fb, truth)
    log(f"stage 2: {len(X2):,} pairs from {len(s1_b):,} S1 records, {len(cols2)} features")
    oof2, model2 = fit_cv(X2, y2, fb["i"].to_numpy(), args, "stage 2")
    model2.save_model(str(cache / "model2.txt"))
    keys_b = fb.select("i", "j").with_columns(pl.Series("p", oof2))
    del fb, X2, y2
    gc.collect()
    scored = with_oof(predict_parquet(model2, f2_path, cols2), keys_b, "p")
    f2_path.unlink()
    gc.collect()
    scored.write_parquet(cache / "train_scored.parquet")   # for error analysis
    truth_b = truth.filter(pl.col("i").is_in(s1_b.implode()))
    log("decision on stage-1 probabilities (for comparison):")
    s1_score, _ = decide.tune(scored.select("i", "j", pl.col("p1").alias("p")), truth_b, s1_b)
    log("decision on stage-2 probabilities:")
    best_score, best_cfg = decide.tune(scored.select("i", "j", "p"), truth_b, s1_b)
    log(f"OOF macro F0.5: stage 1 = {s1_score:.4f}, stage 2 = {best_score:.4f} with {best_cfg}")
    pred = decide.apply(scored.select("i", "j", "p"), best_cfg)
    for c in meta["country"].unique().sort():
        ids = meta.filter((pl.col("country") == c) & pl.col("rid").is_in(s1_b.implode()))["rid"]
        log(f"  country={c:<8} F0.5={decide.macro_f05(pred.filter(pl.col('i').is_in(ids.implode())), truth_b, ids):.4f}")
    (cache / "decision.json").write_text(json.dumps({"cfg": best_cfg, "cols": cols, "cols2": cols2}))
    imp = sorted(zip(model2.feature_importance("gain"), cols2), reverse=True)[:10]
    log("stage-2 top features: " + ", ".join(c for _, c in imp))
    del scored, pred, truth
    gc.collect()

    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "note": args.note, **rep,
           "sample_frac": f"{args.frac_a}/{args.frac_b}", "oof_f05": round(best_score, 4),
           "decision": json.dumps(best_cfg), "lb_public": ""}
    try:
        row["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        row["commit"] = ""

    # ---------------- test
    if not args.skip_test:
        tnorm_path, tcheap_path = build_split("test", args, cache)
        tp1 = score_pairs(tnorm_path, tcheap_path, model1, cols, rates)
        tf2_path = cache / "test_f2.parquet"
        stage2.build(tp1, tcheap_path, tnorm_path).write_parquet(tf2_path)
        del tp1
        gc.collect()
        tscored = predict_parquet(model2, tf2_path, cols2)
        tf2_path.unlink()
        gc.collect()
        tscored.write_parquet(cache / "test_scored.parquet")
        matches = decide.apply(tscored.select("i", "j", "p"), best_cfg)
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
        row["test_cands_per_s1"] = round(len(tscored) / len(ts1), 2)
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
