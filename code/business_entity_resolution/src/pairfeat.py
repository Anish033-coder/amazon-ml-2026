"""Pair features for candidate pairs, computed in chunks so tens of millions of pairs
fit in memory. No feature uses the country label itself."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

# string columns only: token lists are rebuilt per chunk, which keeps memory low
REC_COLS = ["rid", "src", "name_core", "name_full", "name_compact", "name_legal",
            "addr_norm", "addr_numstr", "addr_wordstr", "addr_first_num", "name_was_indic"]


def record_table(norm_path) -> pl.DataFrame:
    """Per-record columns needed for features, plus split-level frequency counts
    (chains share a name across addresses; malls share an address across names)."""
    t = pl.read_parquet(norm_path, columns=REC_COLS).sort("rid")
    return t.with_columns(
        pl.len().over("name_core").log1p().cast(pl.Float32).alias("freq_name"),
        pl.len().over("addr_norm").log1p().cast(pl.Float32).alias("freq_addr"))


def _toks(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _jacc(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(b).list.set_union(pl.col(a)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32)


def cheap_scores(norm_path, pairs: pl.DataFrame, chunk: int = 5_000_000) -> pl.DataFrame:
    """Two fast similarities for *every* candidate pair; used for rank/competition features."""
    t = pl.read_parquet(norm_path, columns=["rid", "name_core", "addr_norm"]).sort("rid")
    name, addr = t["name_core"], t["addr_norm"]
    del t
    outs = []
    for s in range(0, len(pairs), chunk):
        p = pairs.slice(s, chunk)
        i, j = p["i"].to_numpy(), p["j"].to_numpy()
        outs.append(p.with_columns(
            pl.Series("c_name", _cp(name.gather(i).to_list(), name.gather(j).to_list(), fuzz.token_set_ratio)),
            pl.Series("c_addr", _cp(addr.gather(i).to_list(), addr.gather(j).to_list(), fuzz.token_set_ratio))))
    del name, addr
    p = pl.concat(outs).with_columns(((pl.col("c_name") + pl.col("c_addr")) / 2).alias("c_combo"))
    # competition: how does this pair rank for its S1 record, and for its S2/S3 record?
    ex = [pl.len().over("i").cast(pl.UInt16).alias("n_cand_i"), pl.len().over("j").cast(pl.UInt16).alias("n_cand_j")]
    for c in ("n_keys", "c_name", "c_addr", "c_combo"):
        v = pl.col(c).cast(pl.Float32)
        ex += [v.rank("min", descending=True).over("i").cast(pl.Float32).alias(f"{c}_rank_i"),
               (v.max().over("i") - v).alias(f"{c}_gap_i"),
               v.rank("min", descending=True).over("j").cast(pl.Float32).alias(f"{c}_rank_j"),
               (v.max().over("j") - v).alias(f"{c}_gap_j")]
    return p.with_columns(ex)


def full_features(rec: pl.DataFrame, pairs: pl.DataFrame, chunk: int = 2_000_000) -> pl.DataFrame:
    """pairs must carry the cheap_scores columns; returns them plus the full feature set."""
    outs = []
    for s in range(0, len(pairs), chunk):
        p = pairs.slice(s, chunk)
        L = rec[p["i"].to_numpy()]
        R = rec[p["j"].to_numpy()]
        na, nb = L["name_core"].to_list(), R["name_core"].to_list()
        aa, ab = L["addr_norm"].to_list(), R["addr_norm"].to_list()
        f = {
            "n_ratio": _cp(na, nb, fuzz.ratio),
            "n_partial": _cp(na, nb, fuzz.partial_ratio),
            "n_tsort": _cp(na, nb, fuzz.token_sort_ratio),
            "n_jw": _cp(na, nb, JaroWinkler.normalized_similarity),
            "n_lev": _cp(na, nb, Levenshtein.normalized_similarity),
            "n_full_tset": _cp(L["name_full"].to_list(), R["name_full"].to_list(), fuzz.token_set_ratio),
            "n_compact": _cp(L["name_compact"].to_list(), R["name_compact"].to_list(), fuzz.ratio),
            "n_compact_partial": _cp(L["name_compact"].to_list(), R["name_compact"].to_list(), fuzz.partial_ratio),
            "a_ratio": _cp(aa, ab, fuzz.ratio),
            "a_partial": _cp(aa, ab, fuzz.partial_ratio),
            "a_tsort": _cp(aa, ab, fuzz.token_sort_ratio),
            "a_jw": _cp(aa, ab, JaroWinkler.normalized_similarity),
            "w_tset": _cp(L["addr_wordstr"].to_list(), R["addr_wordstr"].to_list(), fuzz.token_set_ratio),
        }
        both = pl.DataFrame({
            "ca": L["name_core"], "cb": R["name_core"], "ta": L["addr_norm"], "tb": R["addr_norm"],
            "ua": L["addr_numstr"], "ub": R["addr_numstr"], "la": L["name_legal"], "lb": R["name_legal"],
            "ha": L["addr_first_num"], "hb": R["addr_first_num"], "aa": L["addr_norm"], "ab": R["addr_norm"],
            "ia": L["name_was_indic"], "ib": R["name_was_indic"]}).with_columns(
                _toks("ca"), _toks("cb"), _toks("ta"), _toks("tb"), _toks("ua"), _toks("ub"))
        g = both.select(
            _jacc("ca", "cb").alias("n_jacc"),
            _jacc("ta", "tb").alias("a_jacc"),
            _jacc("ua", "ub").alias("num_jacc"),
            (pl.col("ca").list.first() == pl.col("cb").list.first()).cast(pl.Float32).alias("n_first_eq"),
            pl.when((pl.col("la") == "") | (pl.col("lb") == "")).then(None)
              .otherwise((pl.col("la") == pl.col("lb")).cast(pl.Float32)).alias("legal_eq"),
            pl.when((pl.col("ha") == "") | (pl.col("hb") == "")).then(None)
              .otherwise((pl.col("ha") == pl.col("hb")).cast(pl.Float32)).alias("house_eq"),
            ((pl.col("aa") == "").cast(pl.Int8) + (pl.col("ab") == "").cast(pl.Int8)).alias("addr_empty"),
            (pl.col("ia") | pl.col("ib")).cast(pl.Int8).alias("indic_any"),
            (pl.col("ca").list.len().cast(pl.Int16) - pl.col("cb").list.len().cast(pl.Int16)).abs().alias("n_len_diff"),
            (pl.col("ua").list.len().cast(pl.Int16) - pl.col("ub").list.len().cast(pl.Int16)).abs().alias("num_len_diff"))
        extra = pl.DataFrame({
            "src3": (R["src"] == 3).cast(pl.Int8),
            "freq_name_i": L["freq_name"], "freq_name_j": R["freq_name"],
            "freq_addr_i": L["freq_addr"], "freq_addr_j": R["freq_addr"]})
        fz = pl.DataFrame(f)
        for c in ("a_ratio", "a_partial", "a_tsort", "a_jw"):
            fz = fz.with_columns(pl.when(g["addr_empty"] > 0).then(None).otherwise(pl.col(c)).alias(c))
        outs.append(pl.concat([p, fz, g, extra], how="horizontal"))
    return pl.concat(outs)


def feature_columns(frame: pl.DataFrame) -> list:
    return [c for c in frame.columns if c not in ("i", "j")]
