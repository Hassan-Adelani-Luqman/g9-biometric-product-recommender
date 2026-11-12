# Project Plan: Multimodal Biometric Product Recommender (Formative 2)

> **Goal:** Build a command-line system that (1) recognises a user's face, (2) runs a product recommendation model for that user, (3) checks the prediction with a voiceprint, and (4) shows the predicted product only if every check passes, following the flow diagram in the brief.
>
> **Due:** Friday, 13 Nov 2026, 23:59 (the submission window closes 18 Nov; 2 attempts allowed). Submit **one PDF** containing the report, the video link, the GitHub link and each member's contributions.

---

## 0. Team & Roles

Fill in names on day one. Each person **owns** one area. Everyone still **contributes biometric data** (images and audio) and **reviews** another member's work.

| Member | Name | Primary ownership | Secondary / review duty |
|---|---|---|---|
| **A** | Hassan Adelani Luqman (`hassan`) | Tabular data: merge, cleaning, EDA, **Product Recommendation Model** | Reviews D's CLI integration |
| **B** | Emma Tiffany Umwari (`emma`) | Image pipeline: collection standards, augmentation, `image_features.csv`, **Facial Recognition Model** | Reviews C's audio pipeline |
| **C** | Parfait Shema Semana (`parfait`) | Audio pipeline: collection standards, visualisation, augmentation, `audio_features.csv`, **Voiceprint Verification Model** | Reviews B's image pipeline |
| **D** | Peter Michael Angelo Rucakibungo (`peter`) | System integration: repo, shared utils, **CLI app**, unauthorised-attempt simulation, final notebook, report and video | Reviews A's merge/model |

**Shared by everyone:**
- Each member records **3+ facial images** (neutral, smiling, surprised) and **2+ audio clips** ("Yes, approve", "Confirm transaction"). Section 2.3 has the conventions.
- Each member writes the report section for their own area and adds their row to the contribution table.
- Each member commits from **their own GitHub account**, so the commit history shows who did what.

---

## 1. How the Rubric Maps to Owners

Use this table as the final QA checklist before submitting. Every criterion is worth 4 pts.

| # | Rubric criterion | What "Exemplary" requires | Owner | Evidence location |
|---|---|---|---|---|
| 1 | EDA Quality & Insight | Summary stats, variable types, **≥3 labelled plots** (distribution, outliers, correlations) **with interpretation** | A | `notebooks/01_data_merge_eda.ipynb`, report §2 |
| 2 | Data Cleaning & Merge Validation | Nulls and duplicates handled, types fixed, **join logic justified**, **post-merge checks shown** | A | notebook 01, `src/data_merge.py`, report §2 |
| 3 | Image Quantity & Diversity | Each member has 3 expressions, with **consistent naming and format** | B (all members supply images) | `data/images/` |
| 4 | Image Augmentation & Feature Extraction | **≥2 augmentations per image**; embeddings/histograms saved in `image_features.csv` | B | notebook 02, `features/image_features.csv` |
| 5 | Audio Quality & Visualisation | 2 phrases per member, clean audio, **waveform + spectrogram plotted and interpreted** | C (all members supply audio) | `data/audio/`, notebook 03 |
| 6 | Audio Augmentation & Feature Extraction | **≥2 augmentations**; **MFCCs + roll-off + energy** saved in `audio_features.csv` | C | notebook 03, `features/audio_features.csv` |
| 7 | Model Implementation | All 3 models implemented, trained and working | A, B, C | `src/train_*.py`, `models/*.joblib` |
| 8 | Evaluation & Multimodal Logic | **Accuracy, F1 and loss for every model**; multimodal logic explained and used | A, B, C (logic: D) | notebook 04, report §5–6 |
| 9 | System Simulation | Full transaction **and** unauthorised demo working smoothly in the CLI | D | `src/app.py`, video |
| 10 | Submission Quality | Report, notebook, code and files are clean, well named and documented | D (everyone writes their part) | README, report PDF |

---

## 2. Conventions (agree on these in Phase 0)

### 2.1 Repository layout
```
g9-biometric-product-recommender/
├── README.md                      # setup, how to run, team, links
├── Plan.md                        # this file
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── customer_social_profiles.csv
│   │   └── customer_transactions.csv
│   ├── processed/
│   │   └── merged_customer_data.csv          # merged + feature-engineered
│   ├── images/
│   │   ├── <member>/<member>_<expression>_<take>.jpg
│   │   └── augmented/                        # generated, may be git-ignored
│   ├── audio/
│   │   ├── <member>/<member>_<phrase>_<take>.wav
│   │   └── augmented/
│   └── unauthorized/                         # non-member face + voice samples
│       ├── faces/
│       └── voices/
├── features/
│   ├── image_features.csv
│   └── audio_features.csv
├── models/
│   ├── face_model.joblib
│   ├── voice_model.joblib
│   ├── product_model.joblib
│   └── *_meta.json                           # label encoders, thresholds, feature lists
├── notebooks/
│   ├── 01_data_merge_eda.ipynb               # Member A
│   ├── 02_image_processing.ipynb             # Member B
│   ├── 03_audio_processing.ipynb             # Member C
│   ├── 04_model_training_evaluation.ipynb    # A, B, C (one section each)
│   └── 05_system_demo.ipynb                  # Member D (CLI run captured in notebook)
├── src/
│   ├── config.py               # paths, member list, member→customer_id map, thresholds
│   ├── data_merge.py           # A
│   ├── image_pipeline.py       # B: load, augment, extract features
│   ├── audio_pipeline.py       # C: load, augment, extract features
│   ├── train_product_model.py  # A
│   ├── train_face_model.py     # B
│   ├── train_voice_model.py    # C
│   └── app.py                  # D: command-line app
└── reports/
    ├── figures/                # every plot saved as PNG for the report
    └── report.pdf
```

### 2.2 Tech stack
- Python 3.10+, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`
- Images: `opencv-python`, `Pillow`, plus `facenet-pytorch` **or** `deepface` for embeddings. If installing these is painful, fall back to `torchvision` ResNet18/MobileNet embeddings, plus colour histograms and HOG from `scikit-image`.
- Audio: `librosa`, `soundfile`, `numpy` (for noise)
- Pin every version in `requirements.txt`. Set `random_state=42` everywhere.

### 2.3 Biometric data collection standards (everyone)
| Item | Standard |
|---|---|
| Member IDs | **Decided (9 Nov): lowercase first names:** `hassan`, `emma`, `parfait`, `peter` (full names in `config.MEMBER_NAMES`) |
| Images | **3 expressions × 2–3 takes** each (`neutral`, `smiling`, `surprised`), `.jpg`, front-facing, face fills about 50% of the frame, even lighting, plain background. Example: `hassan_smiling_01.jpg` |
| Audio | **2 phrases × 2–3 takes** each (`yes_approve`, `confirm_transaction`), `.wav`, **16 kHz mono**, 2–4 s long, quiet room, ~20 cm from the mic, about 0.5 s of silence before speaking. Example: `emma_confirm_transaction_02.wav` |
| Ingest | Raw phone files are normalised with `python src/ingest_biometrics.py <folder>` (renames, converts audio to wav, strips image metadata) |
| Unauthorised samples | At least **2 faces** and **2 voices** from non-members: consenting friends, public-domain images (e.g. a few LFW faces), or TTS voices. Record where each came from in the README. |
| Consent | Members agree to their face and voice being committed to the repo. If the repo is public, note this in the README. |

> Recording **more than the minimum** (2–3 takes) matters. With only 12 images and 8 clips, a train/test split leaves almost nothing to test on.

### 2.4 Git workflow
- `main` is protected. Each member works on `feat/<area>` (e.g. `feat/data-merge`) and opens a PR, which the assigned reviewer approves.
- Notebooks: **one owner per notebook** to avoid `.ipynb` merge conflicts. Clear the outputs of large cells before committing, but keep the final run's outputs in the submitted versions.
- Commit messages use the pattern `area: what changed`, e.g. `audio: add pitch-shift augmentation`.

---

## 3. Phased Plan

### Phase 0: Kick-off & setup · **Sat 7 Nov** · everyone
| Task | Owner |
|---|---|
| Create the GitHub repo and folder skeleton above, `requirements.txt`, `.gitignore`, `README.md` stub; invite all members | D |
| Write `src/config.py` with paths, member list, a **member → customer_id mapping** (Section 4.4) and placeholder thresholds | D |
| Add the two raw CSVs to `data/raw/` | A |
| Confirm naming conventions (2.3), roles (0) and the timeline in a group call | All |
| Everyone sets up an environment and runs `pip install -r requirements.txt` successfully | All |

**Done when:** every member can clone, install and push to a branch.

> **Phase 0 status (7 Nov):** ✅ folder skeleton, ✅ raw CSVs in `data/raw/`, ✅ `src/config.py` (member→customer map verified against the data), ✅ pinned `requirements.txt` (installs cleanly on Python 3.12), ✅ `.gitignore`, ✅ README stub, ✅ local `git init` on `main`.
> **Still open:** create the GitHub repo and push, invite members, fill in member names, group call to confirm conventions, each member installs locally.

---

### Phase 1: Data collection & tabular merge · **Sun 8 – Mon 9 Nov**

#### 1A. Biometric data collection (all members, finish by Sunday night)
- [x] Each member uploads 3 expressions to `data/images/<member>/` (**1 take each so far**; 2–3 recommended)
- [x] Each member uploads 2 phrases to `data/audio/<member>/` (**1 take each so far**)
- [x] Naming, format and quality checks: all 12 images are upright with metadata/GPS stripped; all 8 clips are 16 kHz mono, 2.4–4.2 s, no clipping (Emma's and Parfait's clips are quiet, peak 0.09–0.17, so normalise loudness in Phase 2)
- [ ] D collects unauthorised face and voice samples into `data/unauthorized/`

#### 1B. Data merge & cleaning (Member A)
**What the data looks like (from inspecting the raw files):**
- `customer_social_profiles`: key `customer_id_new` is a string like `A178`. Customers can have **several rows** (one per platform, and sometimes several on the same platform). The last 5 rows are **exact duplicates** of earlier rows (e.g. `A125, LinkedIn, 66, 4.7, Positive`).
- `customer_transactions`: key `customer_id_legacy` is an integer like `178`. There are 150 transactions (IDs 1001–1150). `customer_rating` has **10 missing values**. The 150 transactions come from 75 customers. Only **61 of those customers (117 transactions)** have a social profile; the rest (e.g. 171, 123) have **none**.
- **Target:** `product_category` with 5 classes: Sports, Electronics, Clothing, Groceries, Books.

**Steps:**
1. **Load and inspect:** `.info()`, `.describe()`, dtypes, unique counts, null counts and duplicate counts for each file.
2. **Clean social profiles:**
   - Drop exact duplicate rows and report how many were dropped.
   - Strip and normalise strings (platform, sentiment).
   - Create the join key: `customer_id = int(customer_id_new.lstrip("A"))`.
3. **Clean transactions:**
   - Parse `purchase_date` as datetime and cast numeric columns.
   - Impute missing `customer_rating` with the **per-customer median, falling back to the global median**, and add a `rating_was_missing` flag. Justify this in the notebook.
   - Check for duplicate `transaction_id`s.
   - Check outliers in `purchase_amount` (IQR rule plus a boxplot), and explain whether to keep them.
4. **Aggregate social profiles to one row per customer** *before* joining. This avoids a many-to-many join that would duplicate transactions.
   - `avg_engagement`, `max_engagement`, `avg_purchase_interest`, `n_social_profiles`
   - Platform one-hot counts (`plat_LinkedIn`, `plat_TikTok`, …)
   - Sentiment encoding: proportions (`pct_positive`, `pct_neutral`, `pct_negative`) and a `sentiment_score` (+1/0/−1 average)
5. **Join:** `transactions LEFT JOIN social_agg ON customer_id`.
   - **Justification:** the prediction unit is the *transaction* (what a customer buys). A left join keeps every labelled transaction. Customers without a profile get imputed social features and `has_social_profile = 0`.
   - Also report what an **inner join** would keep (row count) and why we didn't choose it (on a dataset this small, losing rows hurts).
6. **Post-merge validation** (all printed in the notebook):
   - Row count after the join == row count of transactions (proves the join didn't duplicate rows)
   - `transaction_id` is still unique
   - Null counts per column before and after imputation
   - Number and percentage of transactions matched to a social profile
   - Spot-check 2–3 customers by hand against the raw files
7. **Feature engineering:**
   - Date features: `purchase_month`, `purchase_dayofweek`, `is_weekend`
   - Customer history: `cust_txn_count`, `cust_avg_amount`, `cust_prev_category` (most frequent *earlier* category). **Avoid leakage:** compute these from **prior transactions only** (sort by date, expanding window), or leave-one-out.
   - `amount_bucket` (low/med/high)
   - Discuss in the notebook whether `customer_rating` counts as leakage (a rating given after purchase) and decide whether to keep it.
8. Save to `data/processed/merged_customer_data.csv`. Also move the logic into `src/data_merge.py` so the pipeline can be rerun.

> **1B status (8 Nov):** ✅ Done. `python src/data_merge.py` → 150 rows × 35 cols, 0 nulls, 7/7 validation checks pass. Notebook `01_data_merge_eda.ipynb` Part 1 documents every step. Decision: `customer_rating`, `purchase_amount` and `amount_bucket` are flagged as post-purchase (`dm.POST_PURCHASE_COLUMNS`), so the product model should use the `cust_prev_*` history features instead.

#### 1C. EDA (Member A), at least 3 labelled plots, **each followed by 2–3 sentences of interpretation**
- [x] Summary stats table and a variable-type table (numeric, categorical, datetime, ID)
- [x] **Distribution:** histograms/KDE of `purchase_amount`, `engagement_score`, `purchase_interest_score`
- [x] **Outliers:** boxplots of `purchase_amount` by `product_category`
- [x] **Correlations:** heatmap of numeric features
- [x] **Target balance:** bar chart of `product_category` counts (this informs stratified splitting and class weights)
- [x] Bonus: platform vs. category, month vs. category, previous category → next category
- [x] Save every figure to `reports/figures/eda_*.png`

> **1C status (8 Nov):** ✅ Done. 6 EDA figures with interpretation, plus a takeaways table for Phase 3A (§15 of notebook 01). Key findings: the target is near-balanced (baseline 23.3%); no numeric feature separates categories alone; `purchase_month` and `cust_prev_category` carry the most signal; drop redundant pairs (|r| ≥ 0.85) for Logistic Regression.

#### 1D. Pipeline scaffolding (B, C, D in parallel)
- B: `image_pipeline.py` with `load_image`, `augment`, `extract_features` function stubs, and a decision on which embedding library to use
- C: `audio_pipeline.py` with `load_audio`, `augment`, `extract_features` function stubs
- D: `app.py` skeleton with argument parsing and the stage-by-stage flow printing placeholders

**Done when:** all biometric data is committed, `merged_customer_data.csv` exists, the EDA notebook runs top to bottom, and A's PR is approved by D.

---

### Phase 2: Image & audio processing · **Mon 9 – Tue 10 Nov**

#### 2A. Image processing (Member B), notebook `02_image_processing.ipynb`
1. **Load and display** a grid of **every member × 3 expressions** with titles (`hassan – smiling`).
2. **Preprocess:** detect and crop the face (OpenCV Haar cascade or MTCNN), resize to 160×160 (or 224×224), and normalise.
3. **Augment each image with ≥2 transforms; aim for 4–6:**
   - Rotation (±15°), horizontal flip, grayscale (converted back to 3 channels), brightness/contrast jitter, Gaussian blur or noise
   - Show a grid of "original → each augmentation" for at least one image per member.
   - Save augmented images to `data/images/augmented/` named `<original_stem>__<aug>.jpg`.
4. **Extract features** for every original and augmented image:
   - **Embeddings:** FaceNet 512-d (or a 512-d CNN penultimate layer) → columns `emb_0 … emb_511`
   - **Colour histogram:** e.g. 16 bins × 3 channels → `hist_*`
   - Optional: HOG summary or LBP histogram
5. **Write `features/image_features.csv`** with columns:
   `image_id, member, expression, take, augmentation, source_image, emb_0..emb_n, hist_0..hist_m`
   (`source_image` is required later to split data without leakage.)
6. Also extract features for `data/unauthorized/faces/` and label them `member = "unknown"`.
7. Plot a **2-D PCA/t-SNE of the embeddings** coloured by member. This visual shows the classes are separable.

> **2A status (9 Nov):** ✅ Done, except for unauthorized faces, which are waiting on D.
> - **Pipeline and models:** `src/image_pipeline.py` uses OpenCV YuNet (detection) and SFace (128-d embeddings), with no torch needed.
> - **Augmentation:** 8 augmentations per image.
> - **Output:** `image_features.csv` has 108 rows × 194 columns (128 emb + 48 colour hist + 10 LBP), 0 nulls, faces detected in 108/108.
> - **Notebook:** `02_image_processing.ipynb` has 8 figures with interpretation.
> - **Separability:** same-person similarity is always above different-person similarity. Held-out-photo nearest-centroid accuracy is 100% for embeddings vs 71% for histograms, which are confounded by lighting and background.
> - **To do:** rerun `python src/image_pipeline.py` once `data/unauthorized/faces/` is populated.

#### 2B. Audio processing (Member C), notebook `03_audio_processing.ipynb`
1. **Load** every clip with `librosa.load(sr=16000, mono=True)` and trim leading/trailing silence.
2. **Visualise each member's clips:** waveform (`librosa.display.waveshow`) and **mel spectrogram in dB** (`specshow`), in a grid of member × phrase. Under the plots, write **interpretation**: where the speech energy sits, differences in pitch and formants between speakers, background noise level, and how the two phrases differ in duration and structure.
3. **Augment each sample with ≥2 transforms; aim for 3–4:**
   - Pitch shift (±2 semitones), time stretch (0.9× / 1.1×), additive background noise (SNR around 20 dB), time shift, gain change
   - Plot an "original vs. augmented" waveform and spectrogram for one sample.
   - Save to `data/audio/augmented/<stem>__<aug>.wav`.
4. **Extract features** (fixed length per clip, using mean and std over frames):
   - **MFCC 1–13 (or 20), mean and std, plus delta-MFCC mean**
   - **Spectral roll-off**, spectral centroid, spectral bandwidth
   - **RMS energy** (mean and std), zero-crossing rate, chroma mean (optional), duration
5. **Write `features/audio_features.csv`** with columns:
   `audio_id, member, phrase, take, augmentation, source_audio, mfcc_1_mean…, mfcc_1_std…, rolloff_mean, centroid_mean, rms_mean, rms_std, zcr_mean, duration`
6. Extract features for `data/unauthorized/voices/` too, labelled `member = "unknown"`.

#### 2C. Optional combined multimodal feature table (Member D)
- Build `features/multimodal_features.csv`, keyed by `member`, that joins the averaged image and audio features. This shows the "merge and engineer multimodal features" step. It also documents the member → customer_id link used by the product model.

**Done when:** both CSVs exist, contain no NaNs, have the expected row counts (originals × (1 + n_augs)), and C and B have reviewed each other's PRs.

---

### Phase 3: Model training & evaluation · **Tue 10 – Wed 11 Nov**

**Shared rules for all three models (agree on these first):**
- **Split without leakage:** for face and voice, split by `source_image` / `source_audio` using `GroupShuffleSplit` or `StratifiedGroupKFold`, so augmentations of a test image never appear in training.
- Compare **at least 2 algorithms** per model (e.g. Logistic Regression vs. Random Forest vs. XGBoost) and keep the best.
- Report for each model: **Accuracy, macro F1, log-loss**, a confusion matrix and a classification report. Save the figures to `reports/figures/`.
- Save each model with `joblib` to `models/`, along with a `*_meta.json` containing the feature column order, label encoder classes and decision threshold.

#### 3A. Product Recommendation Model (Member A), `train_product_model.py`
- Input: `merged_customer_data.csv`. Target: `product_category`.
- Pipeline: `ColumnTransformer` (scale numeric, one-hot categorical) → classifier.
- Candidates: Logistic Regression (multinomial), Random Forest, XGBoost. Use `class_weight="balanced"` where available.
- Evaluate with **stratified 5-fold CV** (the dataset is small, about 150 rows) plus a held-out test set. Compare against a **majority-class baseline**.
- Show feature importance (RF/XGB) and interpret it.
- **Be honest:** with 5 classes and about 150 rows, accuracy may be modest. Say so in the report, and frame it against the baseline (1/5 = 20% random).
- Provide `predict_for_customer(customer_id) -> (category, probabilities)`, which builds the feature row from that customer's latest data.

> **3A status (10 Nov):** ✅ Done.
> - **Script:** `python src/train_product_model.py` runs in about 1.5 min and saves `models/product_model.joblib` + `product_model_meta.json`.
> - **Selection:** 3 algorithms × 3 feature sets, chosen by repeated-CV log loss on the training split. The winner is **Logistic Regression on `cust_prev_category`** (C = 0.3).
> - **Results:** accuracy is 28.2% in CV, 26.7% on the held-out test and 29.4% on unseen customers. The baselines are 20.8%, 23.3% and 22.7%.
> - **Feature sets:** the all-features models overfit and fall below the baseline on unseen customers, so the social-profile features are not used.
> - **Notebook:** `04_model_training_evaluation.ipynb` Part A has 6 figures (`product_*.png`). The 3D summary table auto-fills from `models/{face,voice}_model_meta.json` once B and C save them.
> - **For D (CLI app):** `from src.train_product_model import predict_for_customer`; `predict_for_customer(187)` returns `("Groceries", {...})`. The four members get four different predictions.

#### 3B. Facial Recognition Model (Member B), `train_face_model.py`
- Input: `image_features.csv`. Classes: the 4 members (+ `unknown` if enough unauthorised samples exist).
- Candidates: Logistic Regression and an SVM/RF on standardised embeddings (+ histograms).
- **Rejecting unknown faces:** set a confidence threshold on `max(predict_proba)`. Optionally add a cosine-distance check against each member's mean embedding. Tune the threshold on validation data so that unauthorised faces are rejected and members are accepted. Save it in `face_model_meta.json`.
- Report accuracy, F1 and log-loss, and the rejection rate on unauthorised faces.

> **3B status (12 Nov):** ✅ Done. Real non-member test: **27/27 rows of 3 non-member photos rejected**. The highest similarity was 0.25, against a threshold of 0.509.
> - **Script:** `python src/train_face_model.py` runs in about 20 s and saves `models/face_model.joblib` + `face_model_meta.json`.
> - **Selection:** leave-one-source-photo-out CV (12 folds, augmented copies held out with their photo), 3 algorithms × 2 feature sets, chosen by log loss. The winner is **Logistic Regression on the 128-d SFace embedding**.
> - **Results:** 100% accuracy and macro F1, log loss 0.007. Every augmentation is recognised.
> - **Rejecting strangers:** classifier confidence cannot do it, because simulated strangers reach 0.89. Instead, **cosine similarity to the member's mean embedding** must be ≥ **0.509**. Genuine scores are ≥ 0.61 and leave-one-member-out strangers ≤ 0.41, so 100% / 100% on simulated data.
> - **For D:** `from src.train_face_model import identify_face`. `identify_face(path)` returns `{"accepted", "member", "similarity", "reason", ...}`.
> - **Non-member photos:** `image_features.csv` now has 135 rows, including 27 labelled `unknown`. They are used for testing only, never for training. Notebook 02, Section 8 shows them.

#### 3C. Voiceprint Verification Model (Member C), `train_voice_model.py`
- Input: `audio_features.csv`. Classes: the 4 members (+ `unknown`).
- Candidates: Logistic Regression, Random Forest, XGBoost on standardised features.
- Use the same threshold approach as the face model, saved in `voice_model_meta.json`.
- Optional: phrase check (was the phrase "Yes, approve" or "Confirm transaction"?) as a secondary classifier. This is not required.
- Report accuracy, F1 and log-loss, and the rejection rate on unauthorised voices.

#### 3D. Consolidated evaluation (A, B, C → `04_model_training_evaluation.ipynb`)
- One section per model, plus a **summary table**:

| Model | Algorithm | Accuracy | Macro F1 | Log-loss | Notes |
|---|---|---|---|---|---|
| Face | | | | | threshold = … |
| Voice | | | | | threshold = … |
| Product | | | | | baseline = … |

**Done when:** all 3 `.joblib` models load in a fresh Python session and predict on one sample, and the summary table is filled in.

---

### Phase 4: System integration & simulation · **Wed 11 – Thu 12 Nov** · Member D (with A, B, C on call)

#### 4.1 Multimodal decision logic (implements the diagram)
```
START
  │
  ▼
[1] Face image → image_pipeline.extract_features → face_model
      ├─ confidence < threshold OR predicted "unknown" → ❌ ACCESS DENIED (face)
      └─ identity = member_x
  │
  ▼
[2] Look up member_x → customer_id (config.MEMBER_TO_CUSTOMER)
    Run product_model.predict_for_customer(customer_id)
    → prediction is computed but HELD (not shown yet)
  │
  ▼
[3] Voice clip → audio_pipeline.extract_features → voice_model
      ├─ confidence < threshold OR "unknown"           → ❌ ACCESS DENIED (voice)
      ├─ voice identity ≠ face identity                → ❌ ACCESS DENIED (identity mismatch)
      └─ voice identity == member_x                    → ✅ APPROVED
  │
  ▼
[4] DISPLAY predicted product + top-3 probabilities
```
- **Cross-modal consistency check:** the voice must belong to *the same person* the face identified. This is the core of the "multimodal logic" criterion, so explain it in the report.

#### 4.2 Member → customer mapping
The team members are not in the transaction data. Map each member to a real `customer_id` from the merged dataset in `config.py`, and document this as a simulation assumption. **Set in Phase 0:** `hassan → 187` (Sports-leaning), `emma → 101` (Books), `parfait → 114` (Clothing), `peter → 120` (Groceries). All four have both transactions and social profiles.

#### 4.3 CLI app (`src/app.py`)
- Non-interactive mode:
  `python src/app.py --face data/images/hassan/hassan_neutral_01.jpg --voice data/audio/hassan/hassan_yes_approve_01.wav`
- Interactive mode: `python src/app.py` prompts for the face path, then the voice path, step by step.
- Demo mode: `python src/app.py --demo` runs every scenario below in sequence (useful for the video).
- Clear stage-by-stage output (`[1/3] Facial recognition… ✅ hassan (0.94)`), timing per step, and graceful handling of missing or invalid files.

#### 4.4 Required test scenarios (all must run without errors)
| # | Face input | Voice input | Expected result |
|---|---|---|---|
| 1 | Member A (authorised) | Member A | ✅ Product displayed |
| 2 | Member B, a *different expression/take* | Member B, a *different phrase* | ✅ Product displayed |
| 3 | **Unauthorised face** | (not reached) | ❌ Denied at face step |
| 4 | Member C | **Unauthorised voice** | ❌ Denied at voice step |
| 5 | Member C | Member D's voice | ❌ Denied: identity mismatch |
| 6 | Augmented image (e.g. rotated) of Member D | Augmented (noisy) voice of Member D | ✅ Robustness demo |

- [ ] D runs every scenario. A, B and C fix their components if a threshold or model misbehaves.
- [ ] Capture the demo run inside `05_system_demo.ipynb` (`!python src/app.py --demo`) so it is visible in the notebook too.

**Done when:** all 6 scenarios behave as expected from a **fresh clone** following the README.

---

### Phase 5: Report, video & submission · **Thu 12 – Fri 13 Nov**

#### 5.1 Report (D compiles; each owner writes their section) → `reports/report.pdf`
1. **Introduction & system overview:** the problem, the flow diagram, the architecture *(D)*
2. **Data merge, cleaning & EDA:** data issues found, cleaning decisions, join justification, post-merge checks, plots with interpretation, engineered features *(A)*
3. **Image pipeline:** collection protocol, sample grid, augmentations, feature design *(B)*
4. **Audio pipeline:** collection protocol, waveforms/spectrograms with interpretation, augmentations, features *(C)*
5. **Models & evaluation:** for each model, algorithm choice, split strategy, metrics table, confusion matrix, thresholds *(A, B, C)*
6. **Multimodal decision logic & system simulation:** the gating logic, cross-modal check, scenario results table, screenshots *(D)*
7. **Limitations & future work:** small dataset, lighting/noise sensitivity, liveness/spoofing, better embeddings *(all)*
8. **Links:** GitHub repo URL and video URL *(D)*
9. **Team contributions table:** member, tasks done, files owned, and an approximate % share *(all)*

#### 5.2 Video (5–7 min, D records; each member presents their part)
- 0:00 Intro and system flow diagram *(D)*
- Data merge and EDA highlights *(A)*
- Image and audio processing highlights *(B, C)*
- Model metrics *(A, B, C)*
- **Live CLI demo:** scenarios 1, 3, 4 and 5 at minimum *(D)*
- Upload as unlisted on YouTube or as a shareable Drive link, and **test the link in a private/incognito window**.

#### 5.3 README.md (D)
Project summary, setup steps, how to rerun each pipeline step in order, how to run the CLI and the demo, the folder guide, team members, and the video link.

#### 5.4 Final QA checklist (everyone, Fri 13 Nov morning)
- [ ] Fresh clone → install → run `data_merge.py` → `train_*.py` → `app.py --demo` works
- [ ] Every notebook runs top to bottom with **no errors** and has its outputs saved
- [ ] `merged_customer_data.csv`, `image_features.csv` and `audio_features.csv` are present and non-empty
- [ ] All images and audio follow the naming convention; each member has 3 expressions and 2 phrases
- [ ] Each model reports accuracy, F1 and log-loss
- [ ] Every rubric row in Section 1 has evidence
- [ ] The report PDF contains the GitHub link, video link and contribution table; both links open while logged out
- [ ] **Submit the PDF by Fri 13 Nov 23:59.** Keep the second attempt in reserve for fixes before 18 Nov.

---

## 4. Timeline at a Glance

| Day | A (Tabular/Product) | B (Image/Face) | C (Audio/Voice) | D (Integration) |
|---|---|---|---|---|
| **Sat 7** | Add raw data, start cleaning | Write image collection guide | Write audio collection guide | Repo, skeleton, config, README |
| **Sun 8** | Merge, validation | Capture own data; QA everyone's images | Capture own data; QA everyone's audio | Collect unauthorised samples; app skeleton |
| **Mon 9** | EDA, feature engineering | Preprocessing + augmentation | Visualisation + augmentation | Shared utils; multimodal feature table |
| **Tue 10** | Product model training | Feature extraction → `image_features.csv`; start face model | Feature extraction → `audio_features.csv`; start voice model | Wire pipelines into app |
| **Wed 11** | Product model eval + `predict_for_customer` | Face model eval + threshold | Voice model eval + threshold | Full integration, scenarios 1–6 |
| **Thu 12** | Write report §2, §5 | Write report §3, §5 | Write report §4, §5 | Bug fixes, demo notebook, report §1/§6, record video |
| **Fri 13** | Final QA | Final QA | Final QA | Compile PDF and **submit** |

**Sync points:** a 15-minute stand-up each evening (what's done, blockers). There are hard hand-offs at the end of **Sun 8** (all raw biometric data in), **Tue 10** (feature CSVs frozen) and **Wed 11** (models frozen).

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Too few images/clips for meaningful train/test splits | Record 2–3 takes per expression/phrase; augment heavily; use grouped CV |
| Data leakage from augmentations into the test set | Split on `source_image` / `source_audio` (grouped split) |
| Product model accuracy is low (small, noisy data) | Compare against a baseline, use stratified CV, discuss honestly. The rubric rewards evaluation, not high accuracy |
| Face-embedding library won't install (e.g. dlib / torch problems) | Fall back to torchvision CNN embeddings, or histograms + HOG |
| Unknown faces or voices get accepted | Tune the confidence threshold and add an `unknown` class from non-member samples |
| Notebook merge conflicts | One owner per notebook; collaborate through `src/` modules |
| Uneven contribution | Ownership table (Section 0), per-member branches/commits, contribution table in the report |
| Last-minute failure | Submit Friday; the second attempt is available until 18 Nov |
