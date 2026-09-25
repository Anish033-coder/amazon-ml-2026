"""Stand-in format check until the official utils/validate_submission.py is available.
Always run the official validator before uploading.

    python tools/check_submission.py --out-dir output --test-dir <dataset>/test
"""
import argparse
import sys
from pathlib import Path


def read_ids(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t")[0] for line in f][1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()
    t = Path(args.test_dir)
    s1 = read_ids(t / "test_source1.tsv")
    right = set(read_ids(t / "test_source2.tsv")) | set(read_ids(t / "test_source3.tsv"))
    problems, lists = [], {}
    for fname, col in (("matching_results.tsv", "matched_entity_ids"),
                       ("candidate_pairs.tsv", "candidate_entity_ids")):
        with open(Path(args.out_dir) / fname, encoding="utf-8") as f:
            lines = f.read().split("\n")
        if lines[0] != f"source1_entity_id\t{col}":
            problems.append(f"{fname}: bad header {lines[0]!r}")
        rows = [ln.split("\t") for ln in lines[1:] if ln != ""]
        seen, d = set(), {}
        for r in rows:
            if len(r) != 2:
                problems.append(f"{fname}: row with {len(r)} fields: {r[:3]}")
                continue
            sid, ids = r[0], [x for x in r[1].split(",") if x]
            if sid in seen:
                problems.append(f"{fname}: duplicate row {sid}")
            seen.add(sid)
            if len(ids) != len(set(ids)):
                problems.append(f"{fname}: duplicate ids in list of {sid}")
            bad = [x for x in ids if x not in right]
            if bad:
                problems.append(f"{fname}: {sid} lists ids not in test S2/S3: {bad[:3]}")
            d[sid] = set(ids)
        missing = set(s1) - seen
        if missing:
            problems.append(f"{fname}: {len(missing)} Source 1 ids missing, e.g. {sorted(missing)[:3]}")
        lists[fname] = d
    m, c = lists["matching_results.tsv"], lists["candidate_pairs.tsv"]
    outside = sum(len(v - c.get(k, set())) for k, v in m.items())
    if outside:
        problems.append(f"{outside} matched ids never appear in candidate_pairs.tsv")
    for p in problems[:30]:
        print("FAIL:", p)
    print("PASS" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
