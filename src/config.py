"""Shared configuration for the biometric product recommender.

Every script and notebook imports paths, member IDs and thresholds from here.
That keeps the team on one set of conventions (see Plan.md, Section 2).
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"
AUGMENTED_IMAGES_DIR = IMAGES_DIR / "augmented"
AUGMENTED_AUDIO_DIR = AUDIO_DIR / "augmented"
UNAUTHORIZED_FACES_DIR = DATA_DIR / "unauthorized" / "faces"
UNAUTHORIZED_VOICES_DIR = DATA_DIR / "unauthorized" / "voices"

FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

SOCIAL_PROFILES_CSV = RAW_DIR / "customer_social_profiles.csv"
TRANSACTIONS_CSV = RAW_DIR / "customer_transactions.csv"
MERGED_CSV = PROCESSED_DIR / "merged_customer_data.csv"
IMAGE_FEATURES_CSV = FEATURES_DIR / "image_features.csv"
AUDIO_FEATURES_CSV = FEATURES_DIR / "audio_features.csv"
MULTIMODAL_FEATURES_CSV = FEATURES_DIR / "multimodal_features.csv"

FACE_MODEL_PATH = MODELS_DIR / "face_model.joblib"
VOICE_MODEL_PATH = MODELS_DIR / "voice_model.joblib"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_model.joblib"
FACE_META_PATH = MODELS_DIR / "face_model_meta.json"
VOICE_META_PATH = MODELS_DIR / "voice_model_meta.json"
PRODUCT_META_PATH = MODELS_DIR / "product_model_meta.json"

# --------------------------------------------------------------------------
# Team members & data-collection conventions
# File names: <member>_<expression>_<take>.jpg  e.g. member_a_smiling_01.jpg
#             <member>_<phrase>_<take>.wav      e.g. member_a_yes_approve_02.wav
# --------------------------------------------------------------------------
MEMBERS = ["member_a", "member_b", "member_c", "member_d"]
UNKNOWN_LABEL = "unknown"

EXPRESSIONS = ["neutral", "smiling", "surprised"]
PHRASES = ["yes_approve", "confirm_transaction"]

IMAGE_EXT = ".jpg"
AUDIO_EXT = ".wav"

# Simulation assumption: team members are not in the transaction data, so
# each one is mapped to a real customer who has both transactions and a
# social profile. The customers were chosen to lean toward different
# categories so the demo shows varied predictions.
MEMBER_TO_CUSTOMER = {
    "member_a": 187,  # 5 transactions, mostly Sports
    "member_b": 101,  # 4 transactions, mostly Books
    "member_c": 114,  # 4 transactions, mostly Clothing
    "member_d": 120,  # 3 transactions, mostly Groceries
}

# --------------------------------------------------------------------------
# Preprocessing parameters
# --------------------------------------------------------------------------
IMAGE_SIZE = (160, 160)   # face crop size fed to the embedding model
SAMPLE_RATE = 16_000      # Hz, mono
N_MFCC = 13

# --------------------------------------------------------------------------
# Decision thresholds (placeholders; tuned in Phase 3 and saved to *_meta.json)
# A prediction is accepted only if max(predict_proba) >= threshold.
# --------------------------------------------------------------------------
FACE_CONFIDENCE_THRESHOLD = 0.60
VOICE_CONFIDENCE_THRESHOLD = 0.60

RANDOM_STATE = 42
