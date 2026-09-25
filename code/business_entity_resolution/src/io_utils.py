"""Reading the challenge TSVs and writing submission files."""
import csv
from pathlib import Path

import pandas as pd

from .normalize import normalize_address, normalize_country, normalize_name

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pd.DataFrame:
    # keep_default_na=False: a business literally called "NA" or "None" must stay a string.
    # QUOTE_NONE: fields are tab-separated; quotes inside names are data, not quoting.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, encoding="utf-8")


def load_sources(data_dir, split: str) -> pd.DataFrame:
    """All three sources of a split stacked into one frame with a `source` column (1/2/3)."""
    frames = []
    for s in (1, 2, 3):
        df = read_tsv(Path(data_dir) / split / f"{split}_source{s}.tsv")
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{split}_source{s}.tsv is missing columns {missing}")
        df = df[COLS].copy()
        df["source"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_ground_truth(data_dir) -> dict:
    gt = read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    return {row.source1_entity_id: {x for x in row.matched_entity_ids.split(",") if x}
            for row in gt.itertuples(index=False)}


def add_normalized(df: pd.DataFrame) -> pd.DataFrame:
    names = pd.DataFrame([normalize_name(x) for x in df["business_name"]], index=df.index)
    addrs = pd.DataFrame([normalize_address(x) for x in df["business_address"]], index=df.index)
    out = pd.concat([df, names, addrs], axis=1)
    out["country_norm"] = [normalize_country(x) for x in df["country"]]
    out["full_text"] = (out["name_core"] + " " + out["addr_full"]).str.strip()
    return out


def write_id_lists(path, s1_ids, mapping: dict, value_col: str):
    """One row per Source 1 id (input order), comma-joined ids, no quoting, empty when none."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{value_col}\n")
        for sid in s1_ids:
            ids = list(dict.fromkeys(mapping.get(sid, [])))
            f.write(f"{sid}\t{','.join(ids)}\n")
