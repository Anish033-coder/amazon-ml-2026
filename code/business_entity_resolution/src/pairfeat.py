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


def token_freqs(norm_path) -> dict:
    """log document frequency of every name token and address word in the split. A word that
    differs between two names is a typo if it is rare, but a different business if it is a
    common word (e.g. 'GTM Federation' vs 'GTM Fetes')."""
    out = {}
    for col, tag in (("core", "name"), ("addr_words", "addr")):
        t = pl.read_parquet(norm_path, columns=[col]).explode(col).drop_nulls(col)
        out[tag] = t.group_by(col).len("n").select(pl.col(col).alias("tok"), pl.col("n").log1p().cast(pl.Float32).alias("lf"))
    return out


def learn_added_word_rates(norm_path, cheap_path, truth: pl.DataFrame, s1_ids: pl.Series,
                           min_n: int = 50) -> pl.DataFrame:
    """Match rate of each word that an S2/S3 name *adds* to its S1 name, learned from the
    training labels of S1 records the models never train on. On train, 'svc'/'ctr' additions
    match ~36% of the time, honorifics ('dr', 'shri') ~57%, and every other added word
    ('grp', 'hldg', 'west', 'tech', ...) 0%: those mark sister businesses."""
    from .normalize import LEGAL
    c = (pl.scan_parquet(cheap_path).select("i", "j", "c_name")
           .filter(pl.col("i").is_in(s1_ids.implode()) & (pl.col("c_name") >= 60)).collect())
    core = pl.read_parquet(norm_path, columns=["rid", "core"])
    c = (c.join(core.rename({"rid": "i", "core": "ci"}), on="i").join(core.rename({"rid": "j", "core": "cj"}), on="j")
          .join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["i", "j"], how="left")
          .with_columns(pl.col("y").fill_null(0),
                        pl.col("cj").list.set_difference("ci").list.eval(
                            pl.element().filter(~pl.element().is_in(list(LEGAL)))).alias("w")))
    w = c.select("w", "y").explode("w").drop_nulls("w")
    return (w.group_by("w").agg(pl.len().alias("n"), pl.col("y").mean().cast(pl.Float32).alias("rate"))
             .filter(pl.col("n") >= min_n).rename({"w": "tok"}))


def _added_word_stats(added: pl.Series, dropped: pl.Series, rates: pl.DataFrame) -> pl.DataFrame:
    """min / max learned match rate of the words an S2/S3 name adds. A word never seen in
    training that is *purely added* (nothing dropped) is treated like the sister qualifiers
    (rate 0), which is what every non-noise added word is on train; when a word is replaced
    instead (possibly a typo) an unseen word stays unknown for the typo features to judge."""
    k = pl.int_range(len(added), dtype=pl.UInt32, eager=True).alias("k")
    pure = pl.DataFrame([k, (dropped.list.len() == 0).alias("pure")])
    e = pl.DataFrame([k, added.alias("tok")]).explode("tok").drop_nulls("tok")
    e = (e.join(rates.select("tok", "rate"), on="tok", how="left").join(pure, on="k")
          .with_columns(pl.when(pl.col("rate").is_null() & pl.col("pure")).then(0.0).otherwise(pl.col("rate")).alias("rate")))
    m = e.group_by("k").agg(pl.col("rate").min().alias("add_rate_min"), pl.col("rate").max().alias("add_rate_max"),
                            pl.col("rate").is_null().sum().cast(pl.Int8).alias("add_unknown"))
    return pl.DataFrame([k]).join(m, on="k", how="left", maintain_order="left").drop("k")


def _freq_stats(lists: pl.Series, freq: pl.DataFrame, tag: str) -> pl.DataFrame:
    """max / min log-frequency of the tokens in each list (null when the list is empty)."""
    k = pl.int_range(len(lists), dtype=pl.UInt32, eager=True).alias("k")
    e = pl.DataFrame([k, lists.alias("tok")]).explode("tok").drop_nulls("tok")
    m = e.join(freq, on="tok", how="left").group_by("k").agg(
        pl.col("lf").max().alias(f"{tag}_fmax"), pl.col("lf").min().alias(f"{tag}_fmin"))
    return pl.DataFrame([k]).join(m, on="k", how="left", maintain_order="left").drop("k")


def _toks(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _toks_from(name: str, series: pl.Series) -> pl.Expr:
    return pl.lit(series).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias(name)


def _jacc(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(b).list.set_union(pl.col(a)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32)


def cheap_scores(norm_path, pairs: pl.DataFrame, cap: int = 40, chunk: int = 5_000_000) -> pl.DataFrame:
    """Two fast similarities for every blocked pair, then keep the `cap` best candidates per S1
    record ranked by similarity + 5 x keys shared (measured on train: F0.5 ceiling 0.9852 at 40,
    vs 0.9844 ranking by keys alone). Finally add rank/competition features."""
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
    p = (pl.concat(outs)
           # an empty address is not evidence against a match: rank those by the name twice
           .with_columns(pl.when(pl.col("c_addr") <= 0).then(2 * pl.col("c_name")).otherwise(pl.col("c_name") + pl.col("c_addr"))
                         .alias("_sim"))
           .with_columns((pl.col("_sim") + 5 * pl.col("n_keys").cast(pl.Float32)).alias("_sc"))
           .sort(["i", "_sc"], descending=[False, True]).group_by("i", maintain_order=True).head(cap)
           .drop("_sc", "_sim").with_columns(((pl.col("c_name") + pl.col("c_addr")) / 2).alias("c_combo")))
    # competition: how does this pair rank for its S1 record, and for its S2/S3 record?
    ex = [pl.len().over("i").cast(pl.UInt16).alias("n_cand_i"), pl.len().over("j").cast(pl.UInt16).alias("n_cand_j")]
    for c in ("n_keys", "c_name", "c_addr", "c_combo"):
        v = pl.col(c).cast(pl.Float32)
        ex += [v.rank("min", descending=True).over("i").cast(pl.UInt16).alias(f"{c}_rank_i"),
               (v.max().over("i") - v).alias(f"{c}_gap_i"),
               v.rank("min", descending=True).over("j").cast(pl.UInt16).alias(f"{c}_rank_j"),
               (v.max().over("j") - v).alias(f"{c}_gap_j")]
    return p.with_columns(ex)


def full_features(rec: pl.DataFrame, pairs: pl.DataFrame, tf: dict, rates: pl.DataFrame = None,
                  chunk: int = 2_000_000) -> pl.DataFrame:
    """pairs must carry the cheap_scores columns; returns them plus the full feature set.
    tf: token_freqs() of the same split; rates: learn_added_word_rates() from train."""
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
            # initials ('GF' for 'GTM Federation') and one-word invented trade names ('Brixavi')
            (((pl.col("cb").list.len() == 1) & (pl.col("ca").list.len() >= 2)
              & (pl.col("ca").list.eval(pl.element().str.slice(0, 1)).list.join("") == pl.col("cb").list.join("")))
             | ((pl.col("ca").list.len() == 1) & (pl.col("cb").list.len() >= 2)
                & (pl.col("cb").list.eval(pl.element().str.slice(0, 1)).list.join("") == pl.col("ca").list.join(""))))
            .cast(pl.Float32).alias("n_acronym"),
            pl.col("ca").list.len().cast(pl.Int8).alias("n_tok_s1"),
            pl.col("cb").list.len().cast(pl.Int8).alias("n_tok_r"),
            pl.when((pl.col("la") == "") | (pl.col("lb") == "")).then(None)
              .otherwise((pl.col("la") == pl.col("lb")).cast(pl.Float32)).alias("legal_eq"),
            pl.when((pl.col("ha") == "") | (pl.col("hb") == "")).then(None)
              .otherwise((pl.col("ha") == pl.col("hb")).cast(pl.Float32)).alias("house_eq"),
            ((pl.col("aa") == "").cast(pl.Int8) + (pl.col("ab") == "").cast(pl.Int8)).alias("addr_empty"),
            (pl.col("ia") | pl.col("ib")).cast(pl.Int8).alias("indic_any"),
            (pl.col("ca").list.len().cast(pl.Int16) - pl.col("cb").list.len().cast(pl.Int16)).abs().alias("n_len_diff"),
            (pl.col("ua").list.len().cast(pl.Int16) - pl.col("ub").list.len().cast(pl.Int16)).abs().alias("num_len_diff"))
        # House-number offset. On training pairs with near-identical name and address the match
        # rate is 97% when numbers are equal, 88% when one contains the other (dropped digit /
        # padding), 73% when they differ by >50 (a typo), but only 6-22% when they differ by
        # 1-50: a sister business a few doors away. Equality alone cannot see this.
        ni = pl.col("ha").str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
        nj = pl.col("hb").str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
        num = both.select(
            (ni - nj).abs().alias("d"),
            ((pl.col("ha") != pl.col("hb")) & (pl.col("ha") != "") & (pl.col("hb") != "")
             & (pl.col("ha").str.contains(pl.col("hb"), literal=True) | pl.col("hb").str.contains(pl.col("ha"), literal=True))).alias("contains"),
            ni.alias("ni"), nj.alias("nj"))
        g = g.with_columns(
            num["d"].cast(pl.Float32).log1p().alias("num_absdiff_log"),
            (num["d"].cast(pl.Float32) / pl.max_horizontal(num["ni"], num["nj"], pl.lit(1)).cast(pl.Float32)).alias("num_reldiff"),
            ((num["d"] > 0) & (num["d"] <= 50)).cast(pl.Float32).alias("num_small_offset"),
            num["contains"].cast(pl.Float32).alias("num_contains"))
        g = g.with_columns(pl.Series("num_str_lev", _cp(L["addr_first_num"].to_list(), R["addr_first_num"].to_list(), Levenshtein.distance)))
        g = g.with_columns(pl.when(pl.col("house_eq").is_null()).then(None).otherwise(pl.col("num_str_lev")).alias("num_str_lev"))
        # what differs between the two records, and is it a typo or a different word?
        dw = both.with_columns(_toks_from("wa", L["addr_wordstr"]), _toks_from("wb", R["addr_wordstr"])).select(
            pl.col("ca").list.set_difference("cb").alias("n_da"), pl.col("cb").list.set_difference("ca").alias("n_db"),
            pl.col("wa").list.set_difference("wb").alias("w_da"), pl.col("wb").list.set_difference("wa").alias("w_db"),
            pl.col("ua").list.set_difference("ub").alias("u_da"), pl.col("ub").list.set_difference("ua").alias("u_db"))
        diff = {}
        for t in ("n", "w"):
            da, db = dw[f"{t}_da"], dw[f"{t}_db"]
            ja, jb = da.list.join(" ").to_list(), db.list.join(" ").to_list()
            empty = (da.list.len() == 0) | (db.list.len() == 0)
            r = _cp(ja, jb, fuzz.ratio)
            r[empty.to_numpy()] = np.nan
            diff[f"{t}_diff_ratio"] = r
            diff[f"{t}_extra_s1"] = da.list.len().cast(pl.Int8).to_numpy()
            diff[f"{t}_extra_r"] = db.list.len().cast(pl.Int8).to_numpy()
        diff["u_extra_s1"] = dw["u_da"].list.len().cast(pl.Int8).to_numpy()
        diff["u_extra_r"] = dw["u_db"].list.len().cast(pl.Int8).to_numpy()
        fz_diff = pl.concat([pl.DataFrame(diff),
                             _freq_stats(dw["n_da"], tf["name"], "n_extra_s1"), _freq_stats(dw["n_db"], tf["name"], "n_extra_r"),
                             _freq_stats(dw["w_da"], tf["addr"], "w_extra_s1"), _freq_stats(dw["w_db"], tf["addr"], "w_extra_r")]
                            + ([_added_word_stats(dw["n_db"], dw["n_da"], rates)] if rates is not None else []),
                            how="horizontal")
        extra = pl.DataFrame({
            "src3": (R["src"] == 3).cast(pl.Int8),
            "freq_name_i": L["freq_name"], "freq_name_j": R["freq_name"],
            "freq_addr_i": L["freq_addr"], "freq_addr_j": R["freq_addr"]})
        fz = pl.DataFrame(f)
        for c in ("a_ratio", "a_partial", "a_tsort", "a_jw"):
            fz = fz.with_columns(pl.when(g["addr_empty"] > 0).then(None).otherwise(pl.col(c)).alias(c))
        outs.append(pl.concat([p, fz, g, fz_diff, extra], how="horizontal"))
    return pl.concat(outs)


def feature_columns(frame: pl.DataFrame) -> list:
    return [c for c in frame.columns if c not in ("i", "j")]
