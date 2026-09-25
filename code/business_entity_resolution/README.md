# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 record, find the matching Source 2 and Source 3 records. The pipeline runs on CPU and uses only the provided training data.

## Pipeline

1. **Normalise** (`src/normalize.py`). Lowercases text, strips accents, and maps `&` to `and`. Also:
   - expands abbreviations and pulls legal forms (`Pvt Ltd`, `LLC`, `SARL`, …) into a separate field
   - folds city and state spellings
   - extracts the postal code, house number and landmark flag

   The same rules apply to every country label.
2. **Blocking** (`src/blocking.py`). Builds TF-IDF nearest neighbours over three text views (name char n-grams, name+address char n-grams, name words). It searches in both directions, S1→S2/S3 (top-k) and S2/S3→S1 (top-k_rev), within the same country label, and takes the union. That union is written to `candidate_pairs.tsv` and is exactly the set the model scores.
3. **Pair features** (`src/features.py`). About 60 country-agnostic features:
   - fuzzy and TF-IDF similarities for name and address
   - postal code and house number agreement
   - how common the name and address are in the split
   - each candidate's rank and score gap within its S1 group and within its S2/S3 group
4. **Model**. LightGBM binary classifier, trained with 5-fold `GroupKFold` grouped by Source 1 id.
5. **Post-processing** (`src/postprocess.py`). Chosen on out-of-fold predictions with the exact macro F0.5 metric. Options:
   - one owner per S2/S3 record
   - per-entity thresholds, or choosing the list that maximises expected F0.5

## Reproduce

```bash
# from this folder, with Python 3.11
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.eda --data-dir <student_resource>/dataset
python -m src.pipeline --data-dir <student_resource>/dataset --out-dir <submission_root>/output --loco
```

Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The run also prints:
- blocking recall ceiling and reduction ratio
- out-of-fold F0.5, overall and per country
- leave-one-country-out F0.5, as a stand-in for the unseen test country

Every run appends a row to `experiments/log.csv`. Random seed: 42.
