"""Facial recognition model.

Owner: Emma Tiffany Umwari

Identifies which team member is in a photo, and rejects faces that belong to
nobody in the team. Input: ``features/image_features.csv`` built by
``image_pipeline.py`` (SFace embeddings + colour/texture histograms, every
source photo with 8 augmented copies).

Evaluation protocol (12 source photos = 4 members x 3 expressions):
    1. Leave-one-source-photo-out CV. Each photo AND its 8 augmented copies
       are predicted by a model that never saw any of them, so augmentation
       cannot leak. Every fold holds out one expression of one member.
    2. 3 algorithms x 2 feature sets are compared on these held-out
       predictions. The combination with the lowest log loss is kept
       (a difference below 0.001 is a tie, won by the smaller feature set).
    3. Rejecting strangers. A 4-class classifier always names one of the 4
       members, so identity is confirmed separately: the face's embedding
       must be close (cosine similarity) to the predicted member's mean
       embedding.
         * Genuine scores: held-out photos vs their own member's mean
           (from step 1).
         * Stranger scores: leave-one-member-out. Each member in turn is
           removed from training and treated as a stranger: a real face
           the model has never seen.
       The threshold is placed in the middle of the gap between the two
       score distributions (or, if they overlap, where balanced accuracy is
       highest).
    4. If ``data/unauthorized/faces/`` has been processed (rows with
       member = "unknown"), their rejection rate is reported too.
    5. The kept model is refit on all member photos and saved for the CLI.

Run from the project root:
    python src/train_face_model.py                       # train, evaluate, save
    python src/train_face_model.py identify <image>      # identify one photo
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src import config
    from src import image_pipeline as ip
except ModuleNotFoundError:
    import config
    import image_pipeline as ip

FEATURE_SETS = {
    "embedding": ip.EMB_COLS,
    "embedding + colour/texture": ip.FEATURE_COLS,
}


def candidates() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        # Platt-scaled probabilities via CalibratedClassifierCV (SVC(probability=True)
        # is deprecated in scikit-learn 1.9).
        "SVM (linear)": make_pipeline(
            StandardScaler(), CalibratedClassifierCV(SVC(kernel="linear", C=1.0),
                                                     method="sigmoid", ensemble=False)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Member rows (training) and unauthorized rows (evaluation only)."""
    df = pd.read_csv(config.IMAGE_FEATURES_CSV)
    unknown = df["member"] == config.UNKNOWN_LABEL
    return df[~unknown].reset_index(drop=True), df[unknown].reset_index(drop=True)


def centroids(emb: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Each member's mean embedding, L2-normalised (their 'enrolled' face)."""
    out = {}
    for member in np.unique(y):
        c = emb[y == member].mean(axis=0)
        out[member] = c / np.linalg.norm(c)
    return out


def similarity(emb: np.ndarray, cents: dict, members) -> np.ndarray:
    """Cosine similarity of each embedding to the given member's centroid."""
    return np.array([e @ cents[m] for e, m in zip(emb, members)])


def scores(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict:
    pred = classes[proba.argmax(axis=1)]
    p = np.clip(proba, 1e-6, 1)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, p / p.sum(axis=1, keepdims=True), labels=classes),
    }


def choose_threshold(genuine: np.ndarray, stranger: np.ndarray) -> float:
    """Middle of the widest threshold range with the best balanced accuracy."""
    cuts = np.unique(np.r_[genuine, stranger])
    mids = np.r_[cuts[0] - 1e-3, (cuts[:-1] + cuts[1:]) / 2, cuts[-1] + 1e-3]
    bal = np.array([((genuine >= t).mean() + (stranger < t).mean()) / 2 for t in mids])
    best = np.flatnonzero(np.isclose(bal, bal.max()))
    # When the classes separate, every cut inside the gap ties: take the gap's middle.
    lo = stranger[stranger < mids[best[0]]].max(initial=mids[best[0]])
    hi = genuine[genuine >= mids[best[-1]]].min(initial=mids[best[-1]])
    return float((lo + hi) / 2) if bal.max() == 1.0 else float(mids[best[len(best) // 2]])


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(verbose: bool = True) -> dict:
    df, unknown = load_data()
    y = df["member"].to_numpy()
    groups = df["source_image"].to_numpy()
    classes = np.unique(y)
    emb = df[ip.EMB_COLS].to_numpy()
    logo = list(LeaveOneGroupOut().split(df, y, groups))
    if verbose:
        print(f"{len(df)} rows from {df['source_image'].nunique()} source photos, "
              f"{len(classes)} members, {len(logo)} folds")

    # 1-2. Algorithm x feature-set comparison on held-out predictions
    oof, rows = {}, []
    for fs_name, cols in FEATURE_SETS.items():
        X = df[cols].to_numpy()
        for algo, model in candidates().items():
            proba = np.zeros((len(y), len(classes)))
            for tr, te in logo:
                fitted = clone(model).fit(X[tr], y[tr])
                proba[np.ix_(te, np.searchsorted(classes, fitted.classes_))] = fitted.predict_proba(X[te])
            name = f"{algo} | {fs_name}"
            oof[name] = proba
            rows.append({"model": name, "algorithm": algo, "feature_set": fs_name,
                         "n_features": len(cols), **scores(y, proba, classes)})
    # Lowest log loss wins; differences below 0.001 count as a tie, and a tie
    # goes to the smaller feature set (fewer features, less to overfit).
    comparison = (pd.DataFrame(rows).set_index("model")
                  .assign(_ll=lambda d: d["log_loss"].round(3))
                  .sort_values(["_ll", "n_features", "accuracy"], ascending=[True, True, False])
                  .drop(columns="_ll"))
    chosen = comparison.index[0]
    algo, fs_name = comparison.loc[chosen, ["algorithm", "feature_set"]]
    proba = oof[chosen]
    pred = classes[proba.argmax(axis=1)]
    if verbose:
        print(comparison.round(4).drop(columns=["algorithm", "feature_set", "n_features"]).to_string())
        print(f"Chosen: {chosen}")

    # 3. Genuine vs simulated-stranger similarity scores
    genuine = np.zeros(len(y))
    for tr, te in logo:
        genuine[te] = similarity(emb[te], centroids(emb[tr], y[tr]), y[te])
    stranger, stranger_member = [], []
    for out in classes:
        inside = y != out
        cents = centroids(emb[inside], y[inside])
        sims = np.column_stack([emb[~inside] @ c for c in cents.values()])
        stranger.extend(sims.max(axis=1))          # the closest enrolled member
        stranger_member.extend([out] * int((~inside).sum()))
    stranger = np.array(stranger)
    threshold = round(choose_threshold(genuine, stranger), 3)

    accepted = (pred == y) & (genuine >= threshold)
    per_aug = (pd.DataFrame({"augmentation": df["augmentation"], "correct": pred == y,
                             "accepted": accepted, "similarity": genuine})
               .groupby("augmentation", sort=False)
               .agg(accuracy=("correct", "mean"), accept_rate=("accepted", "mean"),
                    min_similarity=("similarity", "min")))

    # Final model on every member photo
    X_all = df[FEATURE_SETS[fs_name]].to_numpy()
    final = clone(candidates()[algo]).fit(X_all, y)
    cents = centroids(emb, y)

    # 4. Real unauthorized faces, if any
    unauthorized = None
    if len(unknown):
        u_emb = unknown[ip.EMB_COLS].to_numpy()
        u_pred = final.predict(unknown[FEATURE_SETS[fs_name]].to_numpy())
        u_sim = similarity(u_emb, cents, u_pred)
        unauthorized = {"rows": int(len(unknown)),
                        "source_images": int(unknown["source_image"].nunique()),
                        "reject_rate": round(float((u_sim < threshold).mean()), 4),
                        "max_similarity": round(float(u_sim.max()), 4)}

    result = {
        "df": df, "y": y, "classes": classes, "comparison": comparison, "chosen": chosen,
        "algorithm": algo, "feature_set": fs_name, "oof_proba": proba, "oof_pred": pred,
        "genuine_similarity": genuine, "stranger_similarity": stranger,
        "stranger_member": np.array(stranger_member), "threshold": threshold,
        "genuine_accept_rate": float(accepted.mean()),
        "stranger_reject_rate": float((stranger < threshold).mean()),
        "per_augmentation": per_aug, "unauthorized": unauthorized,
        "final_model": final, "centroids": cents,
    }
    if verbose:
        print(f"Threshold (cosine similarity) = {threshold}: genuine accepted "
              f"{result['genuine_accept_rate']:.1%}, simulated strangers rejected "
              f"{result['stranger_reject_rate']:.1%}")
        print(f"  genuine min {genuine.min():.3f} | stranger max {stranger.max():.3f}")
        print("Unauthorized faces:", unauthorized or "none yet (data/unauthorized/faces/ is empty)")
    return result


def save(result: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result["final_model"],
        "feature_columns": FEATURE_SETS[result["feature_set"]],
        "classes": list(result["classes"]),
        "centroids": result["centroids"],
        "threshold": result["threshold"],
    }
    joblib.dump(bundle, config.FACE_MODEL_PATH)

    m = scores(result["y"], result["oof_proba"], result["classes"])
    cmp_ = result["comparison"]
    meta = {
        "model": "face_recognition",
        "owner": config.MEMBER_NAMES["emma"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": result["algorithm"],
        "feature_set": result["feature_set"],
        "feature_columns": bundle["feature_columns"],
        "classes": bundle["classes"],
        "threshold": result["threshold"],
        "threshold_metric": "cosine similarity between the face embedding and the predicted "
                            "member's mean embedding; accept only if >= threshold",
        "training_rows": int(len(result["df"])),
        "source_images": int(result["df"]["source_image"].nunique()),
        "selection": "algorithm x feature set with the lowest log loss on "
                     "leave-one-source-photo-out predictions (ties within 0.001 go to "
                     "the smaller feature set)",
        "metrics": {
            "held_out_test": {
                "evaluation": "leave-one-source-photo-out: each photo and its augmented "
                              "copies predicted by a model that never saw them",
                "rows": int(len(result["y"])),
                **{k: round(float(v), 4) for k, v in m.items()},
            },
            "candidates": {
                name: {k: round(float(cmp_.loc[name, k]), 4)
                       for k in ["accuracy", "f1_macro", "log_loss"]}
                for name in cmp_.index
            },
            "per_augmentation": {
                aug: {k: round(float(v), 4) for k, v in r.items()}
                for aug, r in result["per_augmentation"].iterrows()
            },
            "rejection": {
                "genuine_accept_rate": round(result["genuine_accept_rate"], 4),
                "simulated_stranger_reject_rate": round(result["stranger_reject_rate"], 4),
                "simulated_stranger_method": "leave-one-member-out: each member removed from "
                                             "training and scored against the other three",
                "genuine_min_similarity": round(float(result["genuine_similarity"].min()), 4),
                "stranger_max_similarity": round(float(result["stranger_similarity"].max()), 4),
                "unauthorized_faces": result["unauthorized"] or "pending: no samples in "
                                                                "data/unauthorized/faces/ yet",
            },
        },
        "confusion_matrix": confusion_matrix(result["y"], result["oof_pred"],
                                             labels=result["classes"]).tolist(),
    }
    config.FACE_META_PATH.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Inference (used by the CLI app)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model() -> dict:
    return joblib.load(config.FACE_MODEL_PATH)


def identify_face(image: str | Path | np.ndarray) -> dict:
    """Who is in this photo? Returns a dict the CLI can print directly.

    ``accepted`` is True only if a face was detected AND its embedding is
    close enough to the predicted member's enrolled face.
    """
    b = load_model()
    img = ip.load_image(image) if not isinstance(image, np.ndarray) else image
    feats = ip.extract_features(img)
    x = np.array([[feats[c] for c in b["feature_columns"]]])
    proba = b["model"].predict_proba(x)[0]
    classes = list(b["model"].classes_)
    member = classes[int(proba.argmax())]
    emb = np.array([feats[c] for c in ip.EMB_COLS])
    sim = float(emb @ b["centroids"][member])

    if not feats["face_detected"]:
        reason = "no face detected"
    elif sim < b["threshold"]:
        reason = (f"face does not match any enrolled member (closest: {member}, "
                  f"similarity {sim:.2f} < {b['threshold']:.2f})")
    else:
        reason = "face recognised"
    return {
        "accepted": reason == "face recognised",
        "member": member if reason == "face recognised" else None,
        "closest_member": member,
        "confidence": round(float(proba.max()), 3),
        "similarity": round(sim, 3),
        "threshold": b["threshold"],
        "reason": reason,
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "identify":
        r = identify_face(sys.argv[2])
        print(("ACCEPTED: " + r["member"]) if r["accepted"] else "ACCESS DENIED", "-", r["reason"])
        print(json.dumps(r, indent=2))
    else:
        res = train_and_evaluate()
        save(res)
        print(f"\nSaved {config.FACE_MODEL_PATH.relative_to(config.ROOT)} "
              f"and {config.FACE_META_PATH.relative_to(config.ROOT)}")
