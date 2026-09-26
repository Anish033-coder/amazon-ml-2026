"""Scalable candidate generation: multi-pass key blocking, run separately per country label.

Each record emits hashed keys built from its *rarest* tokens (rarity = document frequency
within its country). Families:
  compact : whole name without spaces          (catches 'darkbuildwellclinic.com')
  n1      : rarest name token                  (single distinctive word)
  nn      : pairs of rare name tokens          (robust to word order, one typo, suffix noise)
  nnum    : rare name token x house number     (robust to typos in other name tokens)
  numw    : house number x rare address word   (catches renamed / Indic-script names)
  nw      : rare name token x rare address word (catches missing house numbers)
  cnum    : whole name x house number        (common names: the number makes the block small)
  numnum  : pairs of rare numbers            (Indian addresses carry several: '741/3', 'D-61')
  numw6   : house number x 6 rare address words (renamed businesses at the same address)
  ww      : pairs of rare address words      (renamed businesses without a house number)
S1 records use more of their tokens than S2/S3 records (asymmetric keys, see _rare).
  compact_e, nn_e : name-only keys restricted to S2/S3 records whose address is empty,
            with looser block limits (these records have nothing else to match on)
Numbers are reduced to their digits ('2300c' -> '2300') and the rarest ones are used.
A S1 record and a S2/S3 record become candidates when they share a key whose block is
small (<= cap_s1 S1 records and <= cap_r S2/S3 records). Candidates are ranked by how
many keys they share and capped per S1 record.
"""
import polars as pl

FAMILIES = ("compact", "n1", "nn", "nnum", "numw", "nw", "cnum", "numnum", "numw6", "ww", "compact_e", "nn_e",
            "cnum_all", "nnum_all", "pp_e", "frag")
# per-family (cap_s1, cap_r) overrides; others use the build_candidates defaults
FAMILY_CAPS = {"compact_e": (8, 30), "nn_e": (8, 30), "cnum": (10, 60), "cnum_all": (10, 60), "pp_e": (8, 30)}
_SHIFT = 1 << 32


def _hashed_tokens(sub: pl.DataFrame, col: str) -> pl.DataFrame:
    return (sub.select("rid", pl.col(col).alias("tok")).explode("tok").drop_nulls("tok")
               .select("rid", pl.col("tok").hash(seed=11).alias("h")))


def _rare(sub: pl.DataFrame, col: str, k: int, k_s1: int = None) -> pl.DataFrame:
    """rid, h: hashes of the k rarest tokens of each record. S1 records, which are complete
    while their S2/S3 copies are often truncated, keep their k_s1 rarest tokens instead, so a
    word that is rare in a short copy still meets the full record (asymmetric keys)."""
    t = _hashed_tokens(sub, col)
    freq = t.group_by("h").len("df")
    t = (t.join(freq, on="h").sort(["rid", "df"])
          .with_columns(pl.int_range(pl.len()).over("rid").alias("_r")))
    if k_s1 is None:
        return t.filter(pl.col("_r") < k).select("rid", "h")
    lim = sub.select("rid", pl.when(pl.col("src") == 1).then(k_s1).otherwise(k).alias("_k"))
    return t.join(lim, on="rid").filter(pl.col("_r") < pl.col("_k")).select("rid", "h")


def _pairs_key(a, b, seed, ordered=False):
    x = a.join(b, on="rid", suffix="_b")
    if ordered:
        x = x.filter(pl.col("h") < pl.col("h_b"))
    return x.select("rid", pl.struct("h", "h_b").hash(seed=seed).alias("key"))


def _family_keys(sub, fam, rare_name, rare_addr, nums, rare_addr6=None, nums3=None, extra=None):
    seed = 1000 + FAMILIES.index(fam)
    if fam.endswith("_e"):
        if fam == "pp_e":
            return _pairs_key(extra["prefix"], extra["prefix"], seed, ordered=True)
        return _family_keys(sub, fam[:-2], rare_name, rare_addr, nums, rare_addr6, nums3, extra).with_columns(
            pl.col("key").hash(seed=seed))
    if fam == "compact":
        return (sub.filter(pl.col("compact_key").str.len_chars() >= 5)
                   .select("rid", pl.col("compact_key").hash(seed=seed).alias("key")))
    if fam == "n1":
        return rare_name.group_by("rid", maintain_order=True).first().select("rid", pl.col("h").hash(seed=seed).alias("key"))
    if fam == "nn":
        return _pairs_key(rare_name, rare_name, seed, ordered=True)
    if fam == "nnum":
        return _pairs_key(rare_name, nums, seed)
    if fam == "numw":
        return _pairs_key(nums, rare_addr, seed)
    if fam == "nw":
        return _pairs_key(rare_name, rare_addr, seed)
    if fam == "cnum":
        first = nums.group_by("rid", maintain_order=True).first()
        comp = (sub.filter(pl.col("compact_key").str.len_chars() >= 3)
                   .select("rid", pl.col("compact_key").hash(seed=11).alias("h")))
        return _pairs_key(comp, first, seed)
    if fam == "numnum":
        return _pairs_key(nums3, nums3, seed, ordered=True)
    if fam == "numw6":
        return _pairs_key(nums, rare_addr6, seed)
    if fam == "ww":
        return _pairs_key(rare_addr, rare_addr, seed, ordered=True)
    if fam == "frag":       # an exact address fragment with a number ('e 2 8', 'cts 102', '2505 tower 1')
        return extra["frags"].select("rid", pl.col("h").hash(seed=seed).alias("key"))
    if fam == "cnum_all":   # whole name x every number (the shared number need not be the rarest)
        comp = (sub.filter(pl.col("compact_key").str.len_chars() >= 3)
                   .select("rid", pl.col("compact_key").hash(seed=11).alias("h")))
        return _pairs_key(comp, extra["all_nums"], seed)
    if fam == "nnum_all":   # S1 records: every name word x every number; copies: rare words x every number
        return _pairs_key(extra["name_all"], extra["all_nums"], seed)
    raise ValueError(fam)


def _country_candidates(sub, cap_s1, cap_r, max_per_s1, k_name, k_addr, k_num, verbose):
    is_s1 = sub.select("rid", (pl.col("src") == 1).alias("s1"))
    rare_name = _rare(sub, "core_key", k_name, k_s1=6)
    rare_addr = _rare(sub, "addr_words", k_addr, k_s1=8)
    rare_addr6 = _rare(sub, "addr_words", 6, k_s1=10)
    digits = sub.select("rid", pl.col("addr_nums").list.eval(pl.element().str.extract(r"(\d+)")).alias("num_digits"))
    digits = digits.join(sub.select("rid", "src"), on="rid")
    nums = _rare(digits, "num_digits", k_num, k_s1=4)
    nums3 = _rare(digits, "num_digits", 3, k_s1=4)
    all_nums = _rare(digits, "num_digits", 5, k_s1=6)
    name_all = _rare(sub, "core_key", 4, k_s1=8)
    prefix = _rare(sub.with_columns(pl.col("core_key").list.eval(pl.element().str.slice(0, 4))), "core_key", 4, k_s1=6)
    mark = r"\b(?:no|door|hno|h|flat|plot|block|house|shop|office|unit|apt|suite|ste|gali|ward|sector|sec|survey|cts|khasra|kh)\b"
    frags = (sub.select("rid", pl.col("addr_raw").str.to_lowercase().str.split(",").alias("f")).explode("f")
                .with_columns(pl.col("f").str.replace_all(r"[^a-z0-9]+", " ").str.replace_all(mark, " ")
                              .str.replace_all(r"\s+", " ").str.strip_chars())
                .filter(pl.col("f").str.contains(r"\d") & (pl.col("f").str.len_chars() >= 2))
                .select("rid", pl.col("f").hash(seed=13).alias("h")).unique())
    extra = {"all_nums": all_nums, "name_all": name_all, "prefix": prefix, "frags": frags}
    empty_r = sub.filter((pl.col("src") != 1) & (pl.col("addr_words").list.len() == 0)
                         & (pl.col("addr_nums").list.len() == 0)).select("rid")
    codes = []
    for fam in FAMILIES:
        keys = _family_keys(sub, fam, rare_name, rare_addr, nums, rare_addr6, nums3, extra).unique().join(is_s1, on="rid")
        if fam.endswith("_e"):  # S1 side: everyone; S2/S3 side: empty-address records only
            keys = keys.filter(pl.col("s1") | pl.col("rid").is_in(empty_r["rid"].implode()))
        c1, cr = FAMILY_CAPS.get(fam, (cap_s1, cap_r))
        sizes = keys.group_by("key").agg(pl.col("s1").sum().alias("n1"), (~pl.col("s1")).sum().alias("nr"))
        ok = sizes.filter(pl.col("n1").is_between(1, c1) & pl.col("nr").is_between(1, cr)).select("key")
        keys = keys.join(ok, on="key", how="semi")
        code = (keys.filter(pl.col("s1")).select("key", pl.col("rid").alias("i"))
                    .join(keys.filter(~pl.col("s1")).select("key", pl.col("rid").alias("j")), on="key")
                    .select((pl.col("i").cast(pl.UInt64) * _SHIFT + pl.col("j").cast(pl.UInt64)).alias("code")))
        codes.append(code)
        if verbose:
            print(f"    {fam:<8} blocks={len(ok):>10,} pairs={len(code):>11,}", flush=True)
    allc = pl.concat(codes).group_by("code").len("n_keys")
    return (allc.select((pl.col("code") // _SHIFT).cast(pl.UInt32).alias("i"),
                        (pl.col("code") % _SHIFT).cast(pl.UInt32).alias("j"),
                        pl.col("n_keys").cast(pl.UInt16))
                .sort(["i", "n_keys"], descending=[False, True])
                .group_by("i", maintain_order=True).head(max_per_s1))


def build_candidates(df: pl.DataFrame, cap_s1: int = 6, cap_r: int = 40, max_per_s1: int = 30,
                     k_name: int = 4, k_addr: int = 3, k_num: int = 2, verbose=True) -> pl.DataFrame:
    """Returns (i, j, n_keys): i = rid of an S1 record, j = rid of an S2/S3 record."""
    cols = ["rid", "src", "country", "core_key", "compact_key", "addr_nums", "addr_words", "addr_raw"]
    out = []
    for c in df["country"].unique().sort():
        sub = df.filter(pl.col("country") == c).select(cols)
        if verbose:
            print(f"  country={c}: {len(sub):,} records", flush=True)
        out.append(_country_candidates(sub, cap_s1, cap_r, max_per_s1, k_name, k_addr, k_num, verbose))
    return pl.concat(out)


def recall_report(pairs: pl.DataFrame, truth_rid: pl.DataFrame, n_s1: int) -> dict:
    """truth_rid: (i, j) true pairs. Pair recall, candidates per S1, share of S1 records
    whose every true match is a candidate."""
    hit = truth_rid.join(pairs.select("i", "j"), on=["i", "j"], how="semi")
    per = truth_rid.group_by("i").len("t").join(hit.group_by("i").len("h"), on="i", how="left").fill_null(0)
    return {"pair_recall": round(len(hit) / max(len(truth_rid), 1), 4),
            "all_found_share": round(float((per["t"] == per["h"]).mean()), 4),
            "cands_per_s1": round(len(pairs) / max(n_s1, 1), 2)}
