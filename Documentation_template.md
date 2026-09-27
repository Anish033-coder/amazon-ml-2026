# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Null_pointers
**Team Members:** Anish Kumawat (leader), Sushant Verma, Prasoon Patel, Pranav Panmand — IIIT Lucknow
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We link every Source 1 business to its Source 2/3 records with a pipeline that runs on one laptop (CPU, 16 GB RAM):
1. Multi-family key blocking, including MinHash signatures, that keeps 98% of true pairs at about 39 candidates per S1 record.
2. A LightGBM pair classifier on 89 similarity and *difference* features.
3. A second LightGBM that reasons about competition between candidates and agreement among an entity's likely copies.
4. A per-entity decision rule that directly maximises expected macro F0.5.

Key ideas:
- A sister business sits a few doors away. House-number offsets and the added word ("Holding", "Group") separate it from a real copy.
- Indic-script names are transliterated with a dictionary learned from the training labels themselves.
- Every choice was measured before it was kept, on out-of-fold predictions or with a leaderboard probe.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from exploratory analysis of the 2.2M S1 / 10.3M S2+S3 training records:

- **Matches per S1 record:** only **5.6%** of S1 records have no match. The rest have 1–11 matches (mean 3.7), so recall matters as much as precision under per-entity F0.5.
- **Same generator in every country.** The distribution of copies per S1 record is identical in India and the US to three decimals. This became a useful label-free check for France.
- **One owner:** no S2/S3 record matches more than one S1 record. We enforce this at decision time.
- **Sister businesses:** planted look-alikes share the brand name and street but differ in house number (a few doors away) and/or carry an added word ("Hldg", "Grp", "Export"; in France "Groupe", "Développement", "Participations"). Words the generator adds to *real* copies ("Svc", "Ctr", "& Fils", "& Associés") look similar but mean the opposite.
- **Country:** every matched pair shares its country label, so blocking is done within country. Country is never a model feature.
- **Scripts:** about 10% of S2 and 5% of S3 names in India are written in Devanagari, Tamil, Telugu, Kannada or Bengali.
- **Name noise:**
  - moved or changed legal suffixes
  - domain-style and social-handle names (`bauleescoublacamis.com`, `@mobilitesclub`)
  - leetspeak (`C0op`, `5arl`) and misspelt legal words
  - injected accents, typos, transpositions
  - acronyms, and trade names unrelated to the legal name
- **Address noise:**
  - reordering and abbreviations (`R.` = Rue, `Imp` = Impasse)
  - zero-padded or `#` / `N°` numbers
  - French region vs department (`Hauts-de-France` vs `Nord`)
  - missing parts; about 3.4% of S2/S3 addresses are empty
- **France appears only in test,** so nothing about it can be learned from labels.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage gradient-boosted classifier + metric-optimal decision rule
**Core Innovation:**
- house-number offset and word-difference features for sister businesses
- match rates of added words, learned from labels, plus a label-free estimate for words never seen in training
- competition and consensus features in a second-stage model
- expected-F0.5 set selection with a one-owner constraint

---

## 3. Candidate Generation (Blocking)

All text is normalised first:
- lowercase, accent stripping, URL / phone / punctuation removal, leetspeak repair
- small hand-written abbreviation and legal-form tables
- learned Indic → Latin transliteration

Each record then emits hashed keys within its country. Complete S1 records use more of their words than their often truncated copies.

- **Blocking keys used:**
  - whole name without spaces, with and without noise words
  - rarest name word, and pairs of rare name words
  - name word × house number, house number × address word, name word × address word
  - pairs of address words, pairs of numbers
  - address fragments containing a number
  - 4-letter prefixes for copies without an address
  - MinHash / LSH bands over character 3-grams of the name, the name + number and the address, which catch copies where every word has a typo
- **Block limits:** a key is used only if its block is small (default ≤ 6 S1 and ≤ 40 S2/S3 records). Candidates are ranked by name + address similarity and number of shared keys, then capped at 40 per S1.
- **Candidate pairs generated:**
  - train: 86.3M (39.1 per S1)
  - test: 67.9M (39.2 per S1)
- **How we ensured true matches were not lost:** recall was measured on the full training set after every change:
  - pair recall **97.96%**
  - 93.3% of S1 records have *all* their matches in the candidate set
  - F0.5 ceiling of the candidate set **0.9935**
  - `candidate_pairs.tsv` is exactly the set scored by the model

---

## 4. Matching Model

**Features used (stage 1, 89 per pair):**
- **Name features:**
  - RapidFuzz ratio, partial, token-sort and token-set ratio
  - Jaro-Winkler, Levenshtein, compact-name similarity, token Jaccard
  - acronym match, token counts
- **Address features:**
  - fuzzy ratios, word and number Jaccard, empty-address flags
  - house-number equality, offset of the first numbers, and the smallest offset between any two numbers
- **Difference features:**
  - words added or dropped, and whether a changed word is a typo or a different word
  - corpus frequency of the differing words
  - legal-form added / dropped / changed
  - learned match rate of the added word
- **Other:**
  - how many S1 and S2/S3 records carry exactly this name
  - name and address frequency in the split
  - rank and score gap of the pair within its S1 group and its S2/S3 group

**Stage 2 features (25):**
- the stage-1 probability p1, with its rank, gap and share within the S1 group
- its rank within the S2/S3 group and the best competing S1 record's p1
- similarity to the S1 record's most likely other copy
- consensus features: the house number and name shared by the copies stage 1 is most sure about, and whether this candidate agrees with them

**Model type:** LightGBM binary classifiers (MIT license), trained from scratch on the provided data; no pretrained models.
- Stage 1 is trained on 20% of S1 records (17.3M pairs); stage 2 on a different 20%.
- Both use 3-fold `GroupKFold` grouped by S1 record, so every pair stage 2 learns from carries an out-of-fold p1.
- Learning rate 0.05, 127 leaves.

**Threshold selection method:** Per S1 record:
1. The one-owner rule: each S2/S3 record keeps only its best S1.
2. Choose the top-k candidates maximising E[F0.5(k)] ≈ 1.25·Σ_topk p / (0.25·Σ p + k), or predict "no match" when that is more likely.

The calibration exponent and floor are tuned on out-of-fold predictions with the exact metric. For the leaderboard we use a stricter setting (γ = 2.5, floor 0.4), which scored higher on test (0.963 vs 0.961 on v6).

---

## 5. Results & Error Analysis

| Version | Main change | OOF F0.5 | Public LB |
|---|---|---|---|
| v1 | key blocking + 1-stage LightGBM | 0.957 | 0.929 |
| v2 | + second stage | 0.961 | 0.926 |
| v6 | + house-number offsets, wider blocking | 0.977 | 0.963 (strict) |
| v9 | + added-word rates, legal direction, typo repair | 0.981 | 0.9725 (strict) |
| v10 | + audit fixes (noise-free keys, fragment keys, name counts) | 0.9847 | **0.97858** (strict) |
| v11 | + MinHash keys, full-name counts, consensus features, France self-training | 0.9865 | 0.97614 (strict) |
| **v14 (final)** | v11 models, stable label-free French word rates, no self-training | 0.9865 | **0.978992** (strict) |

- **F0.5 Score (macro), validation:** 0.9865 out-of-fold.
- **Per country (validation):** India 0.9846, US 0.9877.
  - France has no labels. We uploaded a probe with the France rows blanked and derived France ≈ 0.944 (v9).
  - Using out-of-fold US/India scores, v10's leaderboard score implies France ≈ 0.95.
- **The v11 lesson:**
  - v11 improved out-of-fold F0.5 but lost on the leaderboard.
  - Links per S1 changed exactly as on train for India and the US, but France gained 5× more links.
  - SHAP contributions on French pairs that flipped from p1 < 0.1 to > 0.9 pointed to the label-free rate of the added word. Our estimator had pushed French descriptor words ("école", "comité", "sportive") across a coarse step, from rate 0 to 0.28, because the larger candidate set changed the statistic.
  - We rebuilt the estimator on near-identical pure additions, where train labels show a clean split (share ≤ 0.6 → 0% match, above → ~93%).
  - Self-training was also dropped: it amplified the error.
- **Common false positives (wrong merges):**
  - a sister business at a nearby house number that shares the brand word
  - chain names with an empty address
- **Common false negatives (missed matches):**
  - France under-links copies with moderately changed names (0.20 per S1 vs 0.31–0.36 in India/US)
  - one descriptor word swapped at the same address, which the model treats as a sister
  - trade names unrelated to the legal name when the address is also truncated
  - Indic-script words never seen in training

---

## 6. Conclusion
Our final submission (v14) scores **0.978992** on the public leaderboard, up from 0.929 for our first submission. The biggest gains came from understanding how the data was generated, not from bigger models. Sister businesses at nearby house numbers, words added to real copies, and fuzzy blocking that catches every copy together moved the score from 0.929 to 0.979. Measuring every idea before trusting it mattered as much: the one statistic we did not re-check after changing the candidate set is the one that cost us a submission.

---

## Appendix

### A. Code Artefacts
Code is in `code/business_entity_resolution/`; see its README for the three reproduction commands.

| File | Role |
|---|---|
| `src/prep.py` | Loading, vectorised normalisation, learned transliteration |
| `src/normalize.py` | Hand-written abbreviation and legal-form tables |
| `src/block.py` | Multi-family key and MinHash blocking, recall report |
| `src/pairfeat.py` | Cheap and full pair features, added-word rates |
| `src/stage2.py` | Competition, sibling and consensus features |
| `src/decide.py` | Exact metric, one-owner rule, expected-F0.5 selection and tuning |
| `src/run.py` | End-to-end orchestration with parquet caching |

Runtime is about 2 hours on an Apple M4 laptop with 16 GB RAM.

**Compliance:**
- no external data, APIs or hosted LLMs
- only LightGBM (MIT) trained on the provided data, plus unsupervised statistics on the test files (permitted)
- permissively licensed libraries only
- original code, written with AI coding assistance (permitted)

### B. Additional Results
- **Blocking ceiling (train, F0.5 if every candidate were decided perfectly):** 0.977 (v1) → 0.985 (v6) → 0.988 (v7) → 0.991 (v10) → 0.9935 (v11, MinHash).
- **Learning curve (stage 1 on a fixed validation set):** 2.5% of S1 records → 0.9772; 5% → 0.9782; 10% → 0.9788. More data helps only slightly, so we trained on 20% for speed.
- **Top stage-2 features:** p1, margin to the best competing S1, rank within the S2/S3 group, share within the S1 group, consensus agreement.
