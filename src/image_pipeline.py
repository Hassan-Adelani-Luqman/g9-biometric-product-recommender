"""Image pipeline: load, detect/align faces, augment, and extract features.

Owner: Emma Tiffany Umwari.

Face models (OpenCV Zoo, ONNX, run by opencv-python; no torch/tensorflow):
  * YuNet  - face detector returning a box, 5 landmarks and a confidence score
  * SFace  - face recogniser producing a 128-d identity embedding from a
             112x112 crop aligned on the landmarks
The models are downloaded on first use into models/pretrained/ and checked
against a SHA-256 hash.

Per image, ``extract_features`` returns:
  * emb_0..emb_127  SFace embedding, L2-normalised (identity)
  * hist_0..hist_47 16-bin colour histogram per B/G/R channel of the aligned
                    face, each channel normalised to sum 1 (skin tone / lighting)
  * lbp_0..lbp_9    uniform LBP texture histogram of the aligned grey face

The same functions serve training (``build_image_features``) and inference
(the CLI app), so a face is processed identically in both.

Run from the project root:
    python src/image_pipeline.py
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.feature import local_binary_pattern

try:
    from src import config
except ModuleNotFoundError:
    import config

# ---------------------------------------------------------------------------
# Pretrained models
# ---------------------------------------------------------------------------
PRETRAINED_DIR = config.MODELS_DIR / "pretrained"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
PRETRAINED = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}

MAX_SIDE = 1024            # images are downscaled so the longest side is <= this
DETECT_THRESHOLD = 0.6     # YuNet confidence needed to accept a face
ALIGNED_SIZE = 112         # SFace input size
HIST_BINS = 16
LBP_POINTS, LBP_RADIUS = 8, 1
N_EMB, N_HIST, N_LBP = 128, 3 * HIST_BINS, LBP_POINTS + 2

EMB_COLS = [f"emb_{i}" for i in range(N_EMB)]
HIST_COLS = [f"hist_{i}" for i in range(N_HIST)]
LBP_COLS = [f"lbp_{i}" for i in range(N_LBP)]
FEATURE_COLS = EMB_COLS + HIST_COLS + LBP_COLS
META_COLS = ["image_id", "member", "expression", "take", "augmentation",
             "source_image", "face_detected", "det_score"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models() -> dict[str, Path]:
    """Download the ONNX models if missing; verify their checksums."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (fname, url, sha) in PRETRAINED.items():
        path = PRETRAINED_DIR / fname
        if not path.exists():
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, path)
        if _sha256(path) != sha:
            path.unlink()
            raise RuntimeError(f"Checksum mismatch for {fname}; deleted it, rerun to download again.")
        paths[key] = path
    return paths


@lru_cache(maxsize=1)
def _detector() -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(str(ensure_models()["yunet"]), "", (320, 320), DETECT_THRESHOLD)


@lru_cache(maxsize=1)
def _recognizer() -> cv2.FaceRecognizerSF:
    return cv2.FaceRecognizerSF.create(str(ensure_models()["sface"]), "")


# ---------------------------------------------------------------------------
# Loading & face detection
# ---------------------------------------------------------------------------
def load_image(path: str | Path, max_side: int = MAX_SIDE) -> np.ndarray:
    """Read an image as BGR uint8, honouring EXIF rotation, longest side <= max_side."""
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


@dataclass
class Face:
    box: tuple[int, int, int, int]   # x, y, w, h
    landmarks: np.ndarray            # 5 x 2: right eye, left eye, nose, mouth corners
    score: float
    raw: np.ndarray                  # YuNet row, needed by SFace.alignCrop


def detect_face(img: np.ndarray) -> Face | None:
    """Return the most confident face, or None if no face passes the threshold."""
    det = _detector()
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    best = faces[np.argmax(faces[:, -1])]
    return Face(
        box=tuple(int(v) for v in best[:4]),
        landmarks=best[4:14].reshape(5, 2),
        score=float(best[-1]),
        raw=best,
    )


def align_face(img: np.ndarray, face: Face) -> np.ndarray:
    """112x112 crop, rotated/scaled so the eyes and mouth land in fixed positions."""
    return _recognizer().alignCrop(img, face.raw)


# ---------------------------------------------------------------------------
# Augmentations (deterministic, so the feature file is reproducible)
# ---------------------------------------------------------------------------
def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _gaussian_noise(img: np.ndarray, sigma: float = 15.0, seed: int = config.RANDOM_STATE) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


AUGMENTATIONS = {
    "rotate_p15": lambda im: _rotate(im, 15),
    "rotate_m15": lambda im: _rotate(im, -15),
    "flip": lambda im: cv2.flip(im, 1),
    "grayscale": lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
    "brighter": lambda im: cv2.convertScaleAbs(im, alpha=1.25, beta=30),
    "darker": lambda im: cv2.convertScaleAbs(im, alpha=0.6, beta=-10),
    "blur": lambda im: cv2.GaussianBlur(im, (9, 9), 0),
    "noise": _gaussian_noise,
}


def augment(img: np.ndarray) -> dict[str, np.ndarray]:
    """Return {"original": img, <aug name>: augmented img, ...}."""
    return {"original": img} | {name: fn(img) for name, fn in AUGMENTATIONS.items()}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def color_histogram(face_bgr: np.ndarray) -> np.ndarray:
    chans = [cv2.calcHist([face_bgr], [c], None, [HIST_BINS], [0, 256]).ravel() for c in range(3)]
    return np.concatenate([h / max(h.sum(), 1) for h in chans])


def lbp_histogram(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(N_LBP + 1))
    return hist / hist.sum()


def extract_features(img: np.ndarray) -> dict:
    """Detect, align and describe the face in ``img``.

    If no face is detected, the whole image is resized to 112x112 and used
    instead, and ``face_detected = 0`` is returned. The row stays usable but
    is clearly marked. The CLI denies access in that case.
    """
    face = detect_face(img)
    if face is not None:
        crop = align_face(img, face)
    else:
        crop = cv2.resize(img, (ALIGNED_SIZE, ALIGNED_SIZE), interpolation=cv2.INTER_AREA)

    emb = _recognizer().feature(crop).ravel().astype(np.float64)
    emb /= np.linalg.norm(emb)

    feats = {
        "face_detected": int(face is not None),
        "det_score": face.score if face is not None else 0.0,
    }
    feats.update(zip(EMB_COLS, emb))
    feats.update(zip(HIST_COLS, color_histogram(crop)))
    feats.update(zip(LBP_COLS, lbp_histogram(crop)))
    return feats


def feature_vector(feats: dict) -> np.ndarray:
    """The model-ready vector, in FEATURE_COLS order."""
    return np.array([feats[c] for c in FEATURE_COLS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------
_NAME_RE = re.compile(r"^(?P<member>[a-z]+)_(?P<expression>[a-z]+)_(?P<take>\d+)$")


def parse_image_name(path: Path) -> dict:
    """'emma_smiling_01.jpg' -> member/expression/take. Unauthorized files -> 'unknown'."""
    if path.parent == config.UNAUTHORIZED_FACES_DIR:
        return {"member": config.UNKNOWN_LABEL, "expression": "unknown", "take": 0}
    m = _NAME_RE.match(path.stem)
    if not m or m["member"] not in config.MEMBERS or m["expression"] not in config.EXPRESSIONS:
        raise ValueError(f"{path.name} does not follow <member>_<expression>_<take>.jpg")
    return {"member": m["member"], "expression": m["expression"], "take": int(m["take"])}


def list_source_images() -> list[Path]:
    members = sorted(config.IMAGES_DIR.glob(f"*/*{config.IMAGE_EXT}"))
    members = [p for p in members if p.parent.name in config.MEMBERS]
    unknown = sorted(
        p for p in config.UNAUTHORIZED_FACES_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return members + unknown


def build_image_features(save_augmented: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Augment every source image, extract features and write image_features.csv."""
    rows = []
    config.AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for path in list_source_images():
        meta = parse_image_name(path)
        img = load_image(path)
        for aug_name, aug_img in augment(img).items():
            image_id = f"{path.stem}__{aug_name}"
            if save_augmented and aug_name != "original":
                cv2.imwrite(str(config.AUGMENTED_IMAGES_DIR / f"{image_id}.jpg"), aug_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append({"image_id": image_id, **meta, "augmentation": aug_name,
                         "source_image": path.stem, **extract_features(aug_img)})

    df = pd.DataFrame(rows)[META_COLS + FEATURE_COLS]
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.IMAGE_FEATURES_CSV, index=False)

    if verbose:
        n_src = df["source_image"].nunique()
        print(f"Source images : {n_src} ({(df['member'] == config.UNKNOWN_LABEL).sum() // (1 + len(AUGMENTATIONS))} unauthorized)")
        print(f"Rows          : {len(df)} = {n_src} x (original + {len(AUGMENTATIONS)} augmentations)")
        print(f"Columns       : {df.shape[1]} ({len(META_COLS)} meta + {N_EMB} emb + {N_HIST} hist + {N_LBP} lbp)")
        print(f"Faces detected: {df['face_detected'].sum()}/{len(df)}")
        print(f"Nulls         : {int(df.isna().sum().sum())}")
        print(f"Saved to      : {config.IMAGE_FEATURES_CSV.relative_to(config.ROOT)}")
    return df


if __name__ == "__main__":
    build_image_features()
