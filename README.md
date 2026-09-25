# Team workspace — Amazon ML Challenge 2026

```
student_resource/            <- organiser download goes here (not in git)
code/business_entity_resolution/
    src/                     <- the pipeline (this is what goes in the zip)
experiments/log.csv          <- one row per run: blocking recall, OOF F0.5, LB score
output/                      <- matching_results.tsv, candidate_pairs.tsv
tools/                       <- format check, zip builder, synthetic smoke data
```

## Setup (Mac and Windows)

Install [uv](https://docs.astral.sh/uv/):
- **Mac:** `brew install uv`
- **Windows:** `winget install --id=astral-sh.uv -e`

Then, from the repo root:

```bash
uv venv --python 3.11 .venv
uv pip install -r code/business_entity_resolution/requirements.txt
```

To activate the environment:
- **Mac:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\activate`

## Run

```bash
cd code/business_entity_resolution
python -m src.eda --data-dir ../../student_resource/dataset
python -m src.pipeline --data-dir ../../student_resource/dataset --out-dir ../../output --loco --note "what changed"
cd ../..
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

To build the final zip: `python tools/make_zip.py --team <team_name>`

## Rules we must not break

- **No external lookups.** No APIs, geocoding, web data, or hosted LLMs (ChatGPT, Claude, …) to label or match records.
- **Licenses.** Any pretrained model must be MIT or Apache-2.0 and at most 8B parameters. Check the model card: Llama and Gemma don't qualify.
- **Trust local CV over the public leaderboard.** Final ranking uses the private split.
- **Log every run.** Add each leaderboard score to `experiments/log.csv`.
