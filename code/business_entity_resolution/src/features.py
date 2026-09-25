"""Pair features. None of them use the country label itself, so they transfer to
an unseen country; they only measure how similar two records are."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

RANK_COLS = ["cos_full_char", "cos_name_char", "n_tset", "a_tset", "combo"]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _pair_cos(M, i, j, chunk=200_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(M[i[s:s + chunk]].multiply(M[j[s:s + chunk]]).sum(1)).ravel()
    return out


def _jaccard(a, b):
    out = np.empty(len(a), np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        x, y = set(x.split()), set(y.split())
        out[k] = len(x & y) / len(x | y) if (x or y) else np.nan
    return out


def _eq_or_nan(a, b):
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    out = (a == b).astype(np.float32)
    out[(a == "") | (b == "")] = np.nan
    return out


def _mask_empty(x, a, b):
    x[(np.asarray(a) == "") | (np.asarray(b) == "")] = np.nan
    return x


def pair_features(df: pd.DataFrame, mats: dict, pairs: pd.DataFrame) -> pd.DataFrame:
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    L, R = df.iloc[i], df.iloc[j]
    f = pd.DataFrame(index=pairs.index)

    na, nb = L["name_core"].to_numpy(), R["name_core"].to_numpy()
    f["n_ratio"] = _cp(na, nb, fuzz.ratio)
    f["n_partial"] = _cp(na, nb, fuzz.partial_ratio)
    f["n_tsort"] = _cp(na, nb, fuzz.token_sort_ratio)
    f["n_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    f["n_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity)
    f["n_lev"] = _cp(na, nb, Levenshtein.normalized_similarity)
    f["n_full_tset"] = _cp(L["name_full"].to_numpy(), R["name_full"].to_numpy(), fuzz.token_set_ratio)
    f["n_compact_ratio"] = _cp(L["name_compact"].to_numpy(), R["name_compact"].to_numpy(), fuzz.ratio)
    f["n_jacc"] = _jaccard(na, nb)
    f["n_first_eq"] = [float(x.split()[:1] == y.split()[:1]) for x, y in zip(na, nb)]
    ia, ib = L["name_initials"].to_numpy(), R["name_initials"].to_numpy()
    ca, cb = L["name_compact"].to_numpy(), R["name_compact"].to_numpy()
    f["n_acronym"] = [float(len(x) > 1 and (x == cy or y == cx)) for x, y, cx, cy in zip(ia, ib, ca, cb)]
    la, lb = np.char.str_len(na.astype(str)), np.char.str_len(nb.astype(str))
    f["n_len_diff"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)
    alt = f["n_tset"].to_numpy().copy()
    aa, ab = L["name_alts"].to_numpy(), R["name_alts"].to_numpy()
    for k in np.nonzero([len(x) > 1 or len(y) > 1 for x, y in zip(aa, ab)])[0]:
        alt[k] = max(fuzz.token_set_ratio(x, y) for x in aa[k] for y in ab[k])
    f["n_alt_tset"] = alt
    f["legal_eq"] = _eq_or_nan(L["name_legal"], R["name_legal"])
    f["legal_missing"] = (L["name_legal"].to_numpy() == "").astype(np.int8) + (R["name_legal"].to_numpy() == "")

    aa_, ab_ = L["addr_full"].to_numpy(), R["addr_full"].to_numpy()
    f["a_ratio"] = _mask_empty(_cp(aa_, ab_, fuzz.ratio), aa_, ab_)
    f["a_partial"] = _mask_empty(_cp(aa_, ab_, fuzz.partial_ratio), aa_, ab_)
    f["a_tsort"] = _mask_empty(_cp(aa_, ab_, fuzz.token_sort_ratio), aa_, ab_)
    f["a_tset"] = _mask_empty(_cp(aa_, ab_, fuzz.token_set_ratio), aa_, ab_)
    f["a_jw"] = _mask_empty(_cp(aa_, ab_, JaroWinkler.normalized_similarity), aa_, ab_)
    f["a_jacc"] = _jaccard(aa_, ab_)
    wa, wb = L["addr_words"].to_numpy(), R["addr_words"].to_numpy()
    f["a_words_tset"] = _mask_empty(_cp(wa, wb, fuzz.token_set_ratio), wa, wb)
    f["nums_jacc"] = _jaccard(L["addr_nums"].to_numpy(), R["addr_nums"].to_numpy())
    f["postal_eq"] = _eq_or_nan(L["addr_postal"], R["addr_postal"])
    f["house_eq"] = _eq_or_nan(L["addr_house"], R["addr_house"])
    f["landmark_any"] = L["addr_landmark"].to_numpy() | R["addr_landmark"].to_numpy()
    f["addr_empty"] = (aa_ == "").astype(np.int8) + (ab_ == "")
    lens_a, lens_b = np.char.str_len(aa_.astype(str)), np.char.str_len(ab_.astype(str))
    f["a_len_ratio"] = np.minimum(lens_a, lens_b) / np.maximum(np.maximum(lens_a, lens_b), 1)

    for view, M in mats.items():
        f[f"cos_{view}"] = _pair_cos(M, i, j)
    f["combo"] = (f["cos_name_char"] + f["cos_full_char"]) / 2

    # How common is this name / address in the split: chains share names across
    # different addresses, malls share addresses across different names.
    for col, tag in (("name_core", "name"), ("addr_full", "addr")):
        counts = df[col].map(df[col].value_counts()).to_numpy()
        f[f"freq_{tag}_i"] = np.log1p(counts[i])
        f[f"freq_{tag}_j"] = np.log1p(counts[j])
    f["src3"] = (R["source"].to_numpy() == 3).astype(np.int8)

    # Context within the candidate graph: is this the best option for the S1 record,
    # and is this S1 record the best option for the S2/S3 record (mutual best)?
    g_i, g_j = pairs["i"], pairs["j"]
    f["n_cand_i"] = g_i.map(g_i.value_counts()).to_numpy()
    f["n_cand_j"] = g_j.map(g_j.value_counts()).to_numpy()
    for col in RANK_COLS:
        v = f[col]
        f[f"{col}_rank_i"] = v.groupby(g_i).rank(ascending=False, method="min")
        f[f"{col}_gap_i"] = v.groupby(g_i).transform("max") - v
        f[f"{col}_rank_j"] = v.groupby(g_j).rank(ascending=False, method="min")
        f[f"{col}_gap_j"] = v.groupby(g_j).transform("max") - v
    return f
