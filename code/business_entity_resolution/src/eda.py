"""The facts that decide the pipeline design. Run first on the real data:

    python -m src.eda --data-dir <student_resource>/dataset
"""
import argparse
from collections import Counter
from pathlib import Path

from .io_utils import load_ground_truth, load_sources, read_tsv
from .metric import macro_f05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    d = Path(args.data_dir)

    for split in ("train", "test"):
        for s in (1, 2, 3):
            p = d / split / f"{split}_source{s}.tsv"
            n_lines = sum(1 for _ in open(p, encoding="utf-8")) - 1
            df = read_tsv(p)
            print(f"{p.name:<22} rows={len(df):>8,} (lines={n_lines:,})  cols={list(df.columns)}")
            if n_lines != len(df):
                print("   !! row count != line count: check quoting / embedded newlines")
            quoted = df.apply(lambda c: c.str.startswith('"')).any(axis=1).sum()
            if quoted:
                print(f"   !! {quoted} rows have a field starting with a double quote")
            print("   country:", dict(Counter(df["country"]).most_common(8)))
            print("   empty name / address:", (df["business_name"] == "").sum(),
                  "/", (df["business_address"] == "").sum())

    train = load_sources(d, "train").set_index("entity_id")
    truth = load_ground_truth(d)
    s1 = train.index[train["source"] == 1]
    truth = {i: truth.get(i, set()) for i in s1}
    sizes = Counter(len(v) for v in truth.values())
    print("\nmatches per S1 entity:", dict(sorted(sizes.items())))
    print(f"singleton rate = {sizes[0] / len(truth):.3f}   all-empty baseline F0.5 = {macro_f05({}, truth):.4f}")

    owners = Counter(r for v in truth.values() for r in v)
    multi = {r: c for r, c in owners.items() if c > 1}
    print(f"S2/S3 ids matched to >1 S1 entity: {len(multi)}  (0 => enforce one owner per record)")
    right = train.index[train["source"] != 1]
    print(f"S2/S3 records matched to some S1: {len(owners)}/{len(right)} = {len(owners) / len(right):.3f}")
    src = Counter(r[:2] for v in truth.values() for r in v)
    print("matched ids by source:", dict(src))

    same_c = [train.at[s, "country"] == train.at[r, "country"] for s, v in truth.items() for r in v]
    print(f"matched pairs sharing the country label: {sum(same_c)}/{len(same_c)}"
          "  (all => safe to block by country)")

    print("\nexamples:")
    shown = 0
    for s, v in truth.items():
        if len(v) >= 2 and shown < 6:
            shown += 1
            print(f"  S1 {train.at[s, 'business_name']!r} | {train.at[s, 'business_address']!r}")
            for r in sorted(v):
                print(f"     {r[:2]} {train.at[r, 'business_name']!r} | {train.at[r, 'business_address']!r}")


if __name__ == "__main__":
    main()
