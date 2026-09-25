"""Load a split and normalise all records with vectorised polars expressions.

Indic-script tokens (Devanagari, Bengali, Tamil, Kannada, ...) are mapped to Latin
with a dictionary learned from the *training ground truth* (an Indic-script S2/S3
name aligned word-by-word with its matched S1 name), falling back to anyascii for
unseen words. Everything else is deterministic hand-written normalisation.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from anyascii import anyascii

from .normalize import ADDR_DROP, ADDR_MAP, LEGAL, NAME_DROP, NAME_MAP, PHRASES

INDIC = r"[ऀ-෿]"
_INDIC_RE = re.compile(INDIC)
_JOINERS = re.compile("[​-‍﻿]")   # zero-width (non-)joiners otherwise split Indic words
_SPLIT_RE = re.compile(r"[^\wऀ-෿]+")

# French abbreviations (the unseen test country). Hand-written, deterministic.
FR_ADDR = {"r": "rue", "av": "ave", "bd": "blvd", "bld": "blvd", "ch": "chemin", "imp": "imp",
           "all": "allee", "rte": "rte", "fbg": "fbg", "pl": "pl", "sq": "sq", "qu": "quai",
           "crs": "cours", "res": "res", "zi": "zi", "za": "za", "zac": "zac"}


def read_source(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def load_split(data_dir, split: str) -> pl.DataFrame:
    frames = []
    for s in (1, 2, 3):
        df = read_source(Path(data_dir) / split / f"{split}_source{s}.tsv")
        frames.append(df.select(
            pl.col("entity_id"), pl.lit(s, pl.UInt8).alias("src"),
            pl.col("business_name").fill_null("").alias("name_raw"),
            pl.col("business_address").fill_null("").alias("addr_raw"),
            pl.col("country").fill_null("").str.strip_chars().str.to_lowercase().alias("country")))
    return pl.concat(frames).with_row_index("rid")


def load_truth_pairs(data_dir) -> pl.DataFrame:
    gt = read_source(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "r_id"}))


# ---------------------------------------------------------------- transliteration
def learn_translit(df: pl.DataFrame, truth: pl.DataFrame, min_count: int = 3, min_share: float = 0.25) -> dict:
    """Indic token -> Latin token, learned from matched train pairs.

    Names: an Indic-script S2/S3 name with the same number of words as its S1 name is
    aligned position by position. Addresses: each Indic address token is mapped to the
    S1 address component it co-occurs with most often (states are written in the
    local script in S2/S3 and in Latin in S1)."""
    s1 = df.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1_id"), "name_raw", "addr_raw")
    r = df.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("r_id"),
                                             pl.col("name_raw").alias("r_name"), pl.col("addr_raw").alias("r_addr"))
    j = (truth.join(r.filter(pl.col("r_name").str.contains(INDIC) | pl.col("r_addr").str.contains(INDIC)), on="r_id")
              .join(s1, on="s1_id"))
    name_votes, addr_votes, addr_tot = defaultdict(Counter), defaultdict(Counter), Counter()
    for rn, ra, sn, sa in j.select("r_name", "r_addr", "name_raw", "addr_raw").iter_rows():
        rn, ra = _JOINERS.sub("", rn), _JOINERS.sub("", ra)
        if _INDIC_RE.search(rn):
            rt = [t for t in _SPLIT_RE.split(rn) if t]
            st = [t for t in _SPLIT_RE.split(sn) if t]
            if len(rt) == len(st):
                for a, b in zip(rt, st):
                    if _INDIC_RE.search(a):
                        name_votes[a][b.lower()] += 1
        if _INDIC_RE.search(ra):
            comps = {c.strip().lower() for c in sa.split(",") if c.strip()}
            for tok in {t for t in _SPLIT_RE.split(ra) if _INDIC_RE.search(t)}:
                addr_tot[tok] += 1
                for c in comps:
                    addr_votes[tok][c] += 1
    out = {}
    for votes, tot in ((addr_votes, addr_tot), (name_votes, None)):
        for tok, c in votes.items():
            best, n = c.most_common(1)[0]
            denom = tot[tok] if tot is not None else sum(c.values())
            # spelling variants (shree / shri / sri) split the vote: keep the most common one
            if n >= min_count and n / denom >= min_share:
                out[tok] = best
    return out


def _translit(text: str, table: dict) -> str:
    if not _INDIC_RE.search(text):
        return text
    text = _JOINERS.sub("", text)
    parts = re.split(r"([\wऀ-෿]+)", text)
    return "".join(table.get(p) or (anyascii(p) if _INDIC_RE.search(p) else p) for p in parts)


# ---------------------------------------------------------------- normalisation
def _clean(e: pl.Expr) -> pl.Expr:
    e = (e.str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()
          .str.replace_all(r"https?://|www\.", " ")
          .str.replace_all(r"\.(?:com|net|org|in|co|fr|biz|info|us)\b", " ")
          .str.replace_all(r"\bn/a\b", " ")
          .str.replace_all(r"\b([a-z])\.", "$1")          # l.l.c. -> llc, p.o. -> po
          .str.replace_all("&", " and ")
          .str.replace_all(r"['’`´]", "")
          .str.replace_all(r"[^\p{L}\p{N}]+", " "))
    for phrase in sorted(PHRASES, key=len, reverse=True):
        e = e.str.replace_all(rf"\b{phrase}\b", PHRASES[phrase])
    return e.str.strip_chars()


def _stem(t: pl.Expr) -> pl.Expr:
    return (pl.when((t.str.len_chars() > 4) & t.str.ends_with("s") & ~t.str.ends_with("ss"))
              .then(t.str.slice(0, t.str.len_chars() - 1)).otherwise(t))


_LEET = str.maketrans("0134578", "oleastb")
_LEGAL_LONG = ["private", "limited", "company", "corporation", "incorporated", "enterprises", "services"]


def _name_typo_map(tokens: pl.Series) -> dict:
    """Name typos seen in the data: digits used as letters ('c0nsultancy', 'specia1ists',
    '5arl' for SARL) and misspelt legal words ('priate', 'pirvate', '1imited'). Word tokens with
    digits are mapped back to letters, and a word within edit distance 1-2 of a long legal word
    is mapped onto it, so typo'd legal forms are recognised as legal forms in every country."""
    from rapidfuzz import process as rfp
    from rapidfuzz.distance import OSA
    vocab = tokens.unique().to_list()
    out = {}
    for t in vocab:
        u = t
        if any(ch.isdigit() for ch in t) and sum(ch.isalpha() for ch in t) >= max(2, len(t) // 2) and not t[0].isdigit() \
                or (t[:1] in "015" and len(t) >= 3 and sum(ch.isalpha() for ch in t) >= len(t) - 1):
            u = t.translate(_LEET)
        if len(u) >= 6 and u not in _LEGAL_LONG:
            # letter swaps count as one edit ('pirvate'); a guard on first and last letters keeps
            # real words such as 'compass' or 'limiter' from being pulled onto a legal word
            hit = rfp.extractOne(u, _LEGAL_LONG, scorer=OSA.distance, score_cutoff=2 if len(u) >= 7 else 1)
            w = hit[0] if hit is not None else ""
            if hit is not None and u[0] == w[0] and (u[-1] == w[-1] or u[-2:] == w[-2:][::-1]):
                u = hit[0]
        if u != t:
            out[t] = u
    out.update({"lnc": "inc"})
    return out


def _tokens(df: pl.DataFrame, col: str, mapping: dict, fix_typos: bool = False) -> pl.DataFrame:
    """rid, pos, tok: cleaned, abbreviation-mapped, stemmed tokens (duplicates removed)."""
    t = (df.select("rid", pl.col(col).str.split(" ").alias("tok"))
           .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))
    if fix_typos:
        t = t.with_columns(pl.col("tok").replace(_name_typo_map(t["tok"])))
    t = (t.with_columns(pl.col("tok").replace(mapping))
           .with_columns(pl.when(pl.col("tok").str.contains(r"^\d+$"))
                           .then(pl.col("tok").str.strip_chars_start("0"))
                           .otherwise(_stem(pl.col("tok"))).alias("tok"))
           .filter(pl.col("tok") != "")
           .unique(["rid", "tok"], keep="first", maintain_order=True))
    return t.with_columns(pl.int_range(pl.len()).over("rid").alias("pos"))


def normalize(df: pl.DataFrame, table: dict) -> pl.DataFrame:
    """Adds name_core, name_full, name_legal, name_compact, name_toks, addr_norm,
    addr_nums, addr_words, addr_first_num, empty flags."""
    has_indic = pl.col("name_raw").str.contains(INDIC) | pl.col("addr_raw").str.contains(INDIC)
    fix = df.filter(has_indic).select(
        "rid",
        pl.col("name_raw").map_elements(lambda s: _translit(s, table), return_dtype=pl.String).alias("n_fix"),
        pl.col("addr_raw").map_elements(lambda s: _translit(s, table), return_dtype=pl.String).alias("a_fix"))
    df = (df.join(fix, on="rid", how="left")
            .with_columns(pl.coalesce("n_fix", "name_raw").alias("n_src"),
                          pl.coalesce("a_fix", "addr_raw").alias("a_src"))
            .drop("n_fix", "a_fix"))
    df = df.with_columns(
        # phone numbers stuck into names ('Cobalt LLC - 5023041431'); names only, since
        # addresses legitimately hold long digit runs ('14200 -14300', '253-45 147')
        _clean(pl.col("n_src").str.replace_all(r"\+?\d[\d\s-]{8,}\d", " ")).alias("n_clean"),
        _clean(pl.col("a_src").str.replace_all(r"(?i)\b(?:p\.?\s?o\.?\s*box|pmb|pob)\s*#?\s*\d+", " ")).alias("a_clean"),
        (pl.col("n_src") != pl.col("name_raw")).alias("name_was_indic"))

    nt = _tokens(df, "n_clean", NAME_MAP, fix_typos=True)
    # rows are already in (rid, pos) order and group_by keeps row order within groups
    name = (nt.group_by("rid").agg(
                pl.col("tok").alias("name_toks"),
                pl.col("tok").filter(pl.col("tok").is_in(list(LEGAL))).sort().alias("legal"),
                pl.col("tok").filter(~pl.col("tok").is_in(list(LEGAL | NAME_DROP | {"dba"}))).alias("core")))
    amap = {**ADDR_MAP, **FR_ADDR}
    at = _tokens(df, "a_clean", amap).filter(~pl.col("tok").is_in(list(ADDR_DROP)))
    addr = (at.group_by("rid").agg(
                pl.col("tok").alias("addr_toks"),
                pl.col("tok").filter(pl.col("tok").str.contains(r"\d")).alias("addr_nums"),
                pl.col("tok").filter(~pl.col("tok").str.contains(r"\d")).alias("addr_words")))
    df = df.join(name, on="rid", how="left").join(addr, on="rid", how="left")
    empty = pl.lit([], pl.List(pl.String))
    df = df.with_columns(
        pl.col(c).fill_null(empty) for c in ("name_toks", "legal", "core", "addr_toks", "addr_nums", "addr_words"))
    df = df.with_columns(
        pl.when(pl.col("core").list.len() > 0).then("core").otherwise("name_toks").alias("core"))
    return df.with_columns(
        pl.col("core").list.join(" ").alias("name_core"),
        pl.col("name_toks").list.join(" ").alias("name_full"),
        pl.col("legal").list.join(" ").alias("name_legal"),
        pl.col("core").list.join("").alias("name_compact"),
        pl.col("addr_toks").list.join(" ").alias("addr_norm"),
        pl.col("addr_words").list.join(" ").alias("addr_wordstr"),
        pl.col("addr_nums").list.join(" ").alias("addr_numstr"),
        pl.col("addr_nums").list.first().fill_null("").alias("addr_first_num"),
    ).drop("n_src", "a_src", "n_clean", "a_clean").sort("rid")


def prepare_split(data_dir, split: str, cache_dir, table: dict = None) -> pl.DataFrame:
    cache = Path(cache_dir) / f"{split}_norm.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    df = load_split(data_dir, split)
    if table is None:
        table = load_or_learn_translit(data_dir, cache_dir, df if split == "train" else None)
    df = normalize(df, table)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df


def load_or_learn_translit(data_dir, cache_dir, train_df=None) -> dict:
    path = Path(cache_dir) / "translit.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if train_df is None:
        train_df = load_split(data_dir, "train")
    table = learn_translit(train_df, load_truth_pairs(data_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, ensure_ascii=False, indent=0), encoding="utf-8")
    return table
