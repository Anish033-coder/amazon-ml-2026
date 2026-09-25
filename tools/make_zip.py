"""Build <team>_submission.zip in the layout the organisers require.

    python tools/make_zip.py --team <team_name>
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "business_entity_resolution"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", default=None, help="filled-in Documentation_template.md (or .pdf)")
    args = ap.parse_args()

    doc = Path(args.doc) if args.doc else next(
        (p for p in (ROOT / "Documentation_template.md", ROOT / "student_resource" / "Documentation_template.md")
         if p.exists()), None)
    files = {
        "output/matching_results.tsv": ROOT / "output" / "matching_results.tsv",
        "output/candidate_pairs.tsv": ROOT / "output" / "candidate_pairs.tsv",
        "code/business_entity_resolution/README.md": CODE / "README.md",
        "code/business_entity_resolution/requirements.txt": CODE / "requirements.txt",
    }
    for p in sorted((CODE / "src").rglob("*.py")):
        files[f"code/business_entity_resolution/src/{p.relative_to(CODE / 'src').as_posix()}"] = p
    if doc:
        files[doc.name] = doc
    missing = [k for k, v in files.items() if not v.exists()]
    if missing or not doc:
        raise SystemExit(f"missing: {missing or ['Documentation_template.md']}")

    out = ROOT / f"{args.team}_submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in files.items():
            z.write(src, arc)
    print(f"wrote {out} ({len(files)} files)")


if __name__ == "__main__":
    main()
