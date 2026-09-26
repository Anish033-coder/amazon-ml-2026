# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 record, find the matching Source 2 and Source 3 records. The pipeline runs on CPU in 16 GB RAM, and uses only the provided training and test files.

## Reproduce

```bash
# Python 3.11, from this folder
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
D=<student_resource>/dataset; C=<cache>; O=<submission>/output
python -m src.run --data-dir $D --cache-dir $C --out-dir $O --lr 0.05 --skip-test   # train
python -m src.run --data-dir $D --cache-dir $C --out-dir $O --stage test-build      # normalise + block test
python -m src.run --data-dir $D --cache-dir $C --out-dir $O --stage predict         # score test, write outputs
```

Running the three steps as separate processes keeps peak memory low. A single call without `--skip-test` does all of it in one go.

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. It takes about 2 hours on an Apple M4 with 16 GB RAM. Intermediate results are cached as parquet in `--cache-dir`. Random seed: 42.

## Pipeline (`src/`)

| Step | File | What it does |
|---|---|---|
| 1. Normalise | `prep.py`, `normalize.py` | See below. |
| 2. Blocking | `block.py` | See below. |
| 3. Features | `pairfeat.py` | About 85 features per pair (see below). No feature uses the country label itself. |
| 4. Stage-2 features | `stage2.py` | See below. |
| 5. Models | `run.py` | Two LightGBM classifiers (stage 1 on pair features, stage 2 on stage-1 probabilities + competition features). `GroupKFold` grouped by S1 record; each trained on a 20% sample of S1 records. |
| 6. Decision | `decide.py` | See below. |

**Normalisation.** Vectorised polars text cleaning, processed in record slices:
- lowercase, strip accents, remove punctuation, URLs, phone numbers in names, the "M/s" prefix
- digits used as letters (`c0nsultancy`, `5arl`) and misspelt legal words (`pirvate`) mapped to their proper form
- hand-written abbreviation and legal-form rules
- state names mapped to codes

Indic-script words (Devanagari, Tamil, Bengali, Kannada, …) are converted with a word dictionary **learned from the training ground truth**: an Indic-script S2/S3 name is aligned word by word with its matched S1 name. `anyascii` is the character-level fallback for unseen words.

**Blocking.** Records emit hashed keys, built within the same country label:
- whole name, rare name words and name-word pairs
- name word × house number, house number × address word, name word × address word
- pairs of address words, pairs of numbers
- exact address fragments that contain a number (`e 2 8`, `cts 102`)
- 4-letter prefixes of name words, for copies without an address

Complete S1 records use more of their words than their (often truncated) S2/S3 copies. Name keys drop noise words that the data adds to real copies (Services, Center, honorifics). A key is used only when its block is small. Candidates are then ranked by name + address similarity and keys shared, and capped at 40 per S1 record. `candidate_pairs.tsv` is exactly the set the models score.

**Features:**
- fuzzy name and address similarities (RapidFuzz)
- differences between names:
  - which words were added or dropped
  - whether a changed word is a typo or a different word, and its corpus frequency
  - the learned match rate of added words
  - legal-suffix changes
- house numbers:
  - equality, and the offset of the first and of the closest pair of numbers (a sister business sits a few doors away)
  - numbers shared
- how many S1 records and S2/S3 records carry exactly this name; how common the name and address are in the split
- competition features: rank and score gap of a pair within its S1 group and its S2/S3 group

**Stage-2 features:**
- the stage-1 probability
- its rank, gap and share within the S1 group
- the best competing S1 for the same S2/S3 record
- agreement with the S1 record's other likely matches

**Decision.** Per S1 record:
- each S2/S3 record keeps only its best S1 (in the training labels no S2/S3 record matches two S1 records)
- then choose the top-k that maximises expected F0.5, or predict "no match"

The settings are chosen on out-of-fold predictions with the exact macro F0.5 metric.

## Compliance with the challenge rules

- **No external data.** No network access, APIs, geocoding, gazetteers, postal datasets or hosted LLMs. The code makes no network calls.
- **Learned only from the provided training files:**
  - the models
  - the transliteration dictionary
  - the added-word match rates
  - the list of noise words
- **Unsupervised statistics on the provided test files,** which the organisers' Q&A permits:
  - token frequencies, name counts and blocking keys
  - label-free rates for added words never seen in training, from the share of pairs with equal house numbers
- **Hand-written rules only.** `normalize.py` contains small hand-written normalisation dictionaries: abbreviations, legal forms, and state name → code. The organisers' Q&A permits these.
- **Model license.** The only models are LightGBM (MIT), trained from scratch. No pretrained weights are used.
- **Libraries.** All libraries are pure-algorithm and permissively licensed (see `requirements.txt`).
- **Original code.** The code is original: nothing is copied from other teams or public solutions. The team wrote it with AI coding assistance, which the organisers' Q&A permits. No AI model is called by the pipeline.
