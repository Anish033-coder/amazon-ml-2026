# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]
**Team Members:** [MEMBER 1 (leader), MEMBER 2, MEMBER 3, MEMBER 4]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We link every Source 1 business to its Source 2/3 records with a pipeline built for CPU and 16 GB RAM:
1. Multi-pass hashed key blocking over the rarest name, number and address tokens.
2. A LightGBM pair classifier on ~70 country-agnostic similarity and *difference* features.
3. A second-stage LightGBM that reasons about competition between candidates.
4. A per-entity decision rule that directly maximises expected macro F0.5.

Key ideas:
- Indic-script names are transliterated with a dictionary learned from the training labels themselves.
- Distinguishing a typo from a different real word defeats the planted look-alike businesses.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from exploratory analysis of the 2.2M S1 / 10.3M S2+S3 training records:

- **Matches per S1 record:** only **5.6%** of S1 records have no match. The rest have 1–11 matches (mean 3.7). Recall therefore matters as much as precision under per-entity F0.5.
- **One owner:** no S2/S3 record matches more than one S1 record. We enforce this at decision time.
- **Unmatched S2/S3 records:** 26% of them. They are mostly one-off look-alikes; 26% of these share an exact name with some S1 record.
- **Country:** every matched pair shares its country label, so blocking is done within country. Country is never a model feature.
- **Scripts:** about 10% of S2 and 5% of S3 names in India are written in Devanagari, Tamil, Telugu, Kannada or Bengali (e.g. `सिटी इंजीनियरिंग प्राइवेट लिमिटेड` = City Engineering Private Limited). State names in addresses are often in the local script.
- **Name noise:**
  - legal suffixes moved or changed (`LLC Moncada …`, `Pvt.` ↔ `Private`)
  - domain-style names (`darkbuildwellclinic.com`)
  - junk tokens (`--`, `<<`, `| www…`, `[Limited]`)
  - injected accents, duplicated words, typos, word transpositions
  - trade names entirely unrelated to the legal name
- **Address noise:**
  - component reordering, abbreviations (`Rd`, `Blvd`, `R.` = Rue)
  - zero-padded or `#`-prefixed numbers, PO Box / PMB additions
  - old city names (`Bombay`), `N/A` / `NULL`
  - missing parts; about 3.4% of S2/S3 addresses are empty
- **Test differs from train.** Test has 5.75 S2/S3 records per S1 versus 4.67 in train, and many planted look-alikes: the same brand word with a different second word ("GTM Fetes" vs "GTM Federation"), or the same name at a different house number.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + metric-optimal decision rule
**Core Innovation:** Transliteration dictionary learned from the labels; typo-vs-different-word difference features; competition features in a second-stage model; expected-F0.5 set selection with a one-owner constraint.

---

## 3. Candidate Generation (Blocking)

All text is normalised first:
- lowercase, accent stripping, URL and punctuation removal
- small hand-written abbreviation and legal-form tables
- learned Indic → Latin transliteration

Each record then emits hashed keys built from its **rarest** tokens (document frequency within its country).

- **Blocking keys used:**
  - whole name without spaces
  - rarest name token
  - pairs of rare name tokens
  - rare name token × house number
  - house number × rare address word
  - rare name token × rare address word
  - name-only keys for S2/S3 records with an empty address, with looser block limits
- **Block limits:** a key is kept only if its block has at most 6 S1 and 40 S2/S3 records. Candidates are ranked by the number of shared keys and capped at 30 per S1.
- **Candidate pairs generated:**
  - train: 55.7M (25.3 per S1), reduction ratio > 99.99%
  - test: 45.0M (25.9 per S1)
- **How we ensured true matches were not lost:**
  - several independent key families, so a record is found through its name *or* its address
  - reverse coverage for records whose name is in another script
  - measured recall on the full training set:
    - pair recall **94.2%**
    - F0.5 ceiling of the candidate set **0.977** (US 0.987, India 0.963)
  - `candidate_pairs.tsv` is exactly the set scored by the model

---

## 4. Matching Model

**Features used (stage 1, ~70 per pair):**
- **Name features:**
  - RapidFuzz ratio, partial ratio, token-sort and token-set ratio
  - Jaro-Winkler and normalised Levenshtein
  - compact-name similarity, token Jaccard, first-token equality
  - legal-form agreement, length difference
- **Address features:**
  - fuzzy ratios, token and number Jaccard, house-number equality
  - word-only similarity, empty-address flags
- **Difference features:**
  - words and numbers present in one record but not the other
  - similarity of those differing words (high = typo, low = different word)
  - their log corpus frequency (rare = typo, common = a different real word)
- **Other:**
  - name and address frequency in the split (chains, shared buildings)
  - whether a name was transliterated
  - source flag
  - competition features: rank and gap of the pair's cheap similarity within its S1 group and within its S2/S3 group

**Stage 2 features:**
- the stage-1 probability p1
- its rank, gap, share and count of strong candidates within the S1 group
- its rank within the S2/S3 group and the best competing S1 record's p1
- similarity of the candidate to the S1 record's most likely other match (true copies agree with each other)

**Model type:** LightGBM binary classifiers (MIT license), trained from scratch; no pretrained models.
- Stage 1: trained on 20% of S1 records.
- Stage 2: trained on a different 20%.
- Both use 3-fold `GroupKFold` grouped by S1 record. Every training pair used by stage 2 carries an out-of-sample p1.

**Threshold selection method:** Per S1 record:
1. The one-owner rule: each S2/S3 record keeps only its best S1.
2. Choose the top-k candidates maximising E[F0.5(k)] ≈ 1.25·Σ_topk p / (0.25·Σ p + k), or predict no match when P(no match) = Π(1 − p) is larger.

The calibration exponent, margin and the alternative fixed-threshold rule are grid-searched on out-of-fold predictions with the exact macro F0.5 metric.

---

## 5. Results & Error Analysis

| Version | Change | Out-of-fold F0.5 | Public LB |
|---|---|---|---|
| v1 | key blocking + 1-stage LightGBM (10% sample) | 0.957 | 0.929 |
| v2 | + second stage (competition, sibling agreement) | [fill] | [fill] |
| v3 | + difference features, better transliteration | [fill] | [fill] |

- **F0.5 Score (macro):** [best validation score]
- **Per country (validation):** India [x], US [x].
  - France has no labels. A leaderboard probe with France predictions blanked estimated France at ≈0.89 against ≈0.94 for US+India in v1.
- **Common false positives (wrong merges):**
  - planted look-alikes that share a brand word ("GTM Fetes" vs "GTM Federation")
  - a sibling branch of the same name at a nearby house number
  - chain names with a missing address
- **Common false negatives (missed matches):**
  - S2/S3 records with an empty address and a common name (ambiguous even for a human)
  - Indic-script names whose words never appeared in the training labels
  - trade names completely unrelated to the legal name when the address is also truncated

---

## 6. Conclusion
[2–3 sentences: final score, what mattered most, lessons.]

---

## Appendix

### A. Code Artefacts
Code is in `code/business_entity_resolution/`. Entry point:

`python -m src.run --data-dir <student_resource>/dataset --out-dir output`

| File | Role |
|---|---|
| `src/prep.py` | Loading, vectorised normalisation, learned transliteration |
| `src/normalize.py` | Hand-written abbreviation and legal-form tables |
| `src/block.py` | Multi-pass key blocking and recall report |
| `src/pairfeat.py` | Cheap and full pair features |
| `src/stage2.py` | Competition and sibling features |
| `src/decide.py` | Exact metric, one-owner rule, expected-F0.5 selection and tuning |
| `src/run.py` | End-to-end orchestration with parquet caching |

Runtime is about 1 hour on an Apple M4 laptop with 16 GB RAM.

**Compliance:**
- no external data, APIs or hosted LLMs
- only a LightGBM model trained on the provided data
- only permissively licensed pure-algorithm libraries
- AI coding assistance, which the organisers permit

### B. Additional Results
[blocking recall by key family, feature importance, ablations]
