# Multimodal Biometric Product Recommender (Group 9)

A command-line system that confirms a user's identity with **facial recognition** and **voiceprint verification**, then shows a **personalised product recommendation**. The recommendation comes from a model trained on merged customer social-media and transaction data.

```
Start → Facial Recognition ──fail──▶ Access Denied
              │ pass
              ▼
      Run Product Recommendation Model (result held)
              │
              ▼
      Voice Validation ──fail / identity mismatch──▶ Access Denied
              │ pass
              ▼
      Display Predicted Product
```

> Status: **Phase 0 (setup) complete.** See [Plan.md](Plan.md) for the full phased plan, roles and timeline.

## Team

| Member | Name | Ownership |
|---|---|---|
| A | Hassan Adelani Luqman | Data merge, EDA, product recommendation model |
| B | Emma Tiffany Umwari | Image pipeline, `image_features.csv`, facial recognition model |
| C | Parfait Shema Semana | Audio pipeline, `audio_features.csv`, voiceprint verification model |
| D | Peter Michael Angelo Rucakibungo | Integration, CLI app, unauthorised-attempt demo, report and video |

## Setup

```bash
git clone <repo-url>
cd g9-biometric-product-recommender
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Repository layout

| Path | Contents |
|---|---|
| `data/raw/` | Original `customer_social_profiles.csv` and `customer_transactions.csv` |
| `data/processed/` | `merged_customer_data.csv` (merged and feature-engineered) |
| `data/images/<member>/` | Facial images, e.g. `hassan_smiling_01.jpg` |
| `data/audio/<member>/` | Voice clips, e.g. `emma_yes_approve_01.wav` (16 kHz mono) |
| `data/unauthorized/` | Non-member faces and voices for the access-denied demo |
| `features/` | `image_features.csv`, `audio_features.csv` |
| `models/` | Trained models (`*.joblib`) plus metadata (`*_meta.json`) |
| `notebooks/` | `01`–`05`, one per pipeline stage |
| `src/` | Pipeline scripts and the CLI app (`app.py`) |
| `reports/` | Figures and the final report |

Shared paths, member IDs, the member → customer mapping and thresholds are in [src/config.py](src/config.py).

## Data collection conventions

- **Images:** 3 expressions (`neutral`, `smiling`, `surprised`) × 2–3 takes, `.jpg`, front-facing, even lighting.
- **Audio:** 2 phrases (`yes_approve`, `confirm_transaction`) × 2–3 takes, `.wav`, 16 kHz mono, 2–4 s, quiet room.
- **Naming:** `<member>_<expression|phrase>_<take>.<ext>`, using member IDs `hassan`, `emma`, `parfait`, `peter`.
- **Consent:** all members agree to their face and voice samples being stored in this repository.

### Adding recordings

Put new recordings in a folder laid out as `Image_features/<Full Name>/<expression>.jpg` and `Audio_features/<Full Name>/<phrase>.m4a` (any common phone format), then run:

```bash
python src/ingest_biometrics.py <folder> --dry-run   # preview the mapping
python src/ingest_biometrics.py <folder>             # copy into data/
```

The script:
- renames files to the convention above, fixing typos such as "suprised";
- converts audio to 16 kHz mono `.wav`;
- applies photo rotation and **strips all image metadata**, including the GPS location phones embed;
- never overwrites existing files. A second recording of the same expression or phrase becomes take `_02`, `_03`, and so on.

## How to run

_To be completed in Phase 4. Planned usage:_

```bash
python src/app.py --face <image.jpg> --voice <clip.wav>   # single transaction
python src/app.py                                         # interactive prompts
python src/app.py --demo                                  # all demo scenarios
```

## Links

- Demo video: _TBD_
- Report: _TBD_
