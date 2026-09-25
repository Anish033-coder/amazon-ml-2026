"""Keep the version history the organisers ask for: archive every uploaded
matching_results.tsv (gzipped) with the commit that produced it, and log its score.

    python tools/record_submission.py --tag v1 --note "key blocking + LGBM"          # before uploading
    python tools/record_submission.py --tag v1 --lb 0.9312                            # after the score appears
"""
import argparse
import csv
import gzip
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments" / "submissions.csv"
FIELDS = ["tag", "time", "commit", "file", "note", "lb_public"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--lb", default="")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(LOG, encoding="utf-8"))) if LOG.exists() else []
    hit = [r for r in rows if r["tag"] == a.tag]
    if hit and a.lb:
        hit[0]["lb_public"] = a.lb
    elif hit:
        raise SystemExit(f"tag {a.tag} already recorded")
    else:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dst = ROOT / "submissions" / f"{a.tag}_matching_results.tsv.gz"
        dst.parent.mkdir(exist_ok=True)
        with open(ROOT / "output" / "matching_results.tsv", "rb") as src, gzip.open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
        subprocess.run(["git", "tag", "-f", f"sub-{a.tag}"], cwd=ROOT, check=True)
        rows.append({"tag": a.tag, "time": time.strftime("%Y-%m-%d %H:%M"), "commit": commit,
                     "file": str(dst.relative_to(ROOT)), "note": a.note, "lb_public": a.lb})
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"recorded {a.tag}")


if __name__ == "__main__":
    main()
