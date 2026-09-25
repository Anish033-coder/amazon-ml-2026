"""Candidate generation: union of TF-IDF nearest neighbours over several text views,
searched in both directions (S1 -> S2/S3 and S2/S3 -> S1)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# view name -> (column, vectorizer kwargs)
VIEWS = {
    "name_char": ("name_core", dict(analyzer="char_wb", ngram_range=(2, 4))),
    "full_char": ("full_text", dict(analyzer="char_wb", ngram_range=(3, 3))),
    "name_word": ("name_core", dict(analyzer="word", token_pattern=r"\S+")),
    "addr_char": ("addr_full", dict(analyzer="char_wb", ngram_range=(3, 3))),
}
BLOCK_VIEWS = ("name_char", "full_char", "name_word")


def fit_views(df: pd.DataFrame) -> dict:
    """TF-IDF matrix per view for every record of the split (rows aligned with df)."""
    mats = {}
    for view, (col, kw) in VIEWS.items():
        vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **kw)
        mats[view] = vec.fit_transform(df[col].where(df[col] != "", "_empty_")).tocsr()
    return mats


def _topk(A, B, k: int, min_sim: float):
    """For each row of A, the k most cosine-similar rows of B (rows are L2-normalised)."""
    k = min(k, B.shape[0])
    if k == 0 or A.shape[0] == 0:
        return np.empty(0, int), np.empty(0, int)
    BT = B.T.tocsc()
    chunk = max(1, int(2e7 // B.shape[0]))
    rows, cols = [], []
    for start in range(0, A.shape[0], chunk):
        S = (A[start:start + chunk] @ BT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sims = np.take_along_axis(S, idx, axis=1)
        keep = sims > min_sim
        rows.append(np.nonzero(keep)[0] + start)
        cols.append(idx[keep])
    return np.concatenate(rows), np.concatenate(cols)


def build_candidates(df: pd.DataFrame, mats: dict, k_fwd: int = 10, k_rev: int = 3,
                     by_country: bool = True, min_sim: float = 0.05) -> pd.DataFrame:
    """Returns unique (i, j) pairs: i = row of a Source 1 record, j = row of a S2/S3 record."""
    is_s1 = (df["source"] == 1).to_numpy()
    groups = df["country_norm"].to_numpy() if by_country else np.zeros(len(df), dtype=object)
    pairs = []
    for g in pd.unique(groups[is_s1]):
        left = np.nonzero(is_s1 & (groups == g))[0]
        right = np.nonzero(~is_s1 & (groups == g))[0]
        if len(right) == 0:
            continue
        for view in BLOCK_VIEWS:
            A, B = mats[view][left], mats[view][right]
            r, c = _topk(A, B, k_fwd, min_sim)
            pairs.append(np.stack([left[r], right[c]], 1))
            r, c = _topk(B, A, k_rev, min_sim)
            pairs.append(np.stack([left[c], right[r]], 1))
    if not pairs:
        return pd.DataFrame({"i": [], "j": []}, dtype=int)
    out = pd.DataFrame(np.concatenate(pairs), columns=["i", "j"]).drop_duplicates()
    return out.sort_values(["i", "j"]).reset_index(drop=True)


def candidates_by_s1(df: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    ids = df["entity_id"].to_numpy()
    out = {}
    for i, j in zip(pairs["i"].to_numpy(), pairs["j"].to_numpy()):
        out.setdefault(ids[i], []).append(ids[j])
    return out
