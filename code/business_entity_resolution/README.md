# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 record, find the matching Source 2 and Source 3 records. The pipeline runs on CPU in 16 GB RAM, and uses only the provided training and test files.

## Reproduce

```bash
# Python 3.11, from this folder
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.run --data-dir <student_resource>/dataset --cache-dir <cache> --out-dir <submission>/output
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. It takes about 20 minutes on an Apple M4 with 16 GB RAM. Intermediate results are cached as parquet in `--cache-dir`. Random seed: 42.

## Pipeline (`src/`)

| Step | File | What it does |
|---|---|---|
| 1. Normalise | `prep.py`, `normalize.py` | See below. |
| 2. Blocking | `block.py` | Records emit hashed keys from their rarest tokens: whole name, name-token pairs, name token × house number, house number × address word, name token × address word, plus name-only keys for S2/S3 records with an empty address. Keys are built within the same country label. An S1 and an S2/S3 record become candidates when they share a small block. Candidates are capped at 30 per S1 record, ranked by keys shared. |
| 3. Features | `pairfeat.py` | About 50 features per pair (see below). No feature uses the country label itself. |
| 4. Model | `run.py` | LightGBM binary classifier, `GroupKFold` grouped by S1 record, trained on a sample of S1 records. |
| 5. Decision | `decide.py` | See below. |

**Normalisation.** Vectorised polars text cleaning:
- lowercase, strip accents, remove punctuation and URLs
- hand-written abbreviation and legal-form rules
- state names mapped to codes

Indic-script words (Devanagari, Tamil, Bengali, Kannada, …) are converted with a word dictionary **learned from the training ground truth**: an Indic-script S2/S3 name is aligned word by word with its matched S1 name. `anyascii` is the character-level fallback for unseen words.

**Features:**
- fuzzy name and address similarities (RapidFuzz)
- token, number and house-number agreement
- name and address frequency in the split, to catch chains and shared buildings
- competition features: the rank and score gap of a pair within its S1 group and within its S2/S3 group

**Decision.** Per S1 record:
- each S2/S3 record keeps only its best S1 (in the training labels no S2/S3 record matches two S1 records)
- then choose the top-k that maximises expected F0.5, or predict "no match"

The settings are chosen on out-of-fold predictions with the exact macro F0.5 metric.

## Compliance with the challenge rules

- **No external data.** No network access, APIs, geocoding, gazetteers, postal datasets or hosted LLMs. The code makes no network calls.
- **Only provided data.** The model and the transliteration dictionary are learned only from the provided training files. Unsupervised statistics (token frequencies, blocking keys) are computed on the provided test files, which the organisers' Q&A permits.
- **Hand-written rules only.** `normalize.py` contains small hand-written normalisation dictionaries: abbreviations, legal forms, and state name → code. The organisers' Q&A permits these.
- **Model license.** The only model is LightGBM (MIT), trained from scratch. No pretrained weights are used.
- **Libraries.** All libraries are pure-algorithm and permissively licensed (see `requirements.txt`).
- **Development.** The team wrote this code with AI coding assistance, which the organisers' Q&A permits. No AI model is called by the pipeline.
