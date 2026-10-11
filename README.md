# Business Entity Resolution — Amazon ML Challenge 2026

**Team Null_pointers** (IIIT Lucknow): Anish Kumawat (lead), Sushant Verma, Prasoon Patel, Pranav Panmand

Business records arrive from three independent sources with no shared ID, and with typos, truncated names, missing addresses and names written in Indic scripts. The task: for every Source 1 record, find all the Source 2 and Source 3 records that describe the same real business, without merging look-alike "sister" businesses. Scored with macro F0.5, which punishes a wrong merge more than a missed match.

**Result:** **0.978992** F0.5 on the public leaderboard, up from 0.929 for our first submission. 0.9865 out-of-fold on 2.2M training records. Runs on one laptop CPU with 16 GB RAM, using no external data, APIs or pretrained models.

## How it works

1. **Normalise** names and addresses: accents, punctuation, digits used as letters (`c0nsultancy`), misspelt legal words, abbreviations. Indic-script words are transliterated with a dictionary learned from the training labels.
2. **Block**: generate candidate pairs from name, address and house-number keys plus MinHash signatures. Keeps 98% of true pairs at about 39 candidates per record.
3. **Score**: a LightGBM classifier on 89 similarity and *difference* features (which words were added or dropped, house-number offsets, how common the name is).
4. **Re-score**: a second LightGBM that looks at competing candidates and at agreement among a record's likely copies.
5. **Decide**: per record, pick the set of matches that maximises expected F0.5, or predict "no match".

| Version | Main change | Public LB |
|---|---|---|
| v1 | key blocking + one LightGBM | 0.929 |
| v6 | + house-number offsets, wider blocking | 0.963 |
| v10 | + audit fixes to keys and name counts | 0.97858 |
| v11 | + MinHash keys, consensus features | 0.97614 |
| **v14 (final)** | fixed the v11 regression (found with SHAP) | **0.978992** |

v11 scored better on validation but worse on the leaderboard. SHAP on the pairs that flipped showed that a statistic for French words, which appear only in test, had shifted when the candidate set grew. The write-up explains how we found and fixed it.

## Read more

- [Methodology write-up](Documentation_template.md): data analysis, blocking, features, error analysis
- [Pipeline README](code/business_entity_resolution/README.md): every step in detail, exact commands to reproduce, rules compliance
- [`experiments/`](experiments/): one row per run and per leaderboard submission

## Repository layout

```
code/business_entity_resolution/
    src/              the pipeline (normalise, block, features, models, decision)
    requirements.txt  pinned, permissively licensed libraries
experiments/          run log and leaderboard submissions
tools/                submission checks, zip builder, synthetic data, experiments
Documentation_template.md   methodology write-up submitted to the challenge
```

The challenge dataset is not included. It belongs to the organisers. To reproduce, place their `student_resource/` download in the repo root and follow the [pipeline README](code/business_entity_resolution/README.md). It takes about 2 hours on an Apple M4.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install -r code/business_entity_resolution/requirements.txt
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```
