"""Voiceprint verification model.

Owner: Parfait Shema Semana

Checks that a voice clip belongs to the member the face model recognised.
Input: ``Audio_features/data/audio_features.csv``, built by Parfait's
``Audio_features/audio_pipeline.py`` (64 MFCC / spectral / energy / pitch
statistics per clip; every recording with 5 augmented copies). The same
``load_clip`` and ``extract_features`` functions are used here at inference,
so a clip is processed identically in training and in the CLI.

The voice step is a *verification* task: the face step already claims an
identity, and the voice must confirm it. A clip is accepted for member m only
if the classifier's top prediction is m AND P(m | clip) >= threshold.

Evaluation protocol (8 recordings = 4 members x 2 phrases, 1 take each):
    1. Leave-one-recording-out CV. Each recording AND its augmented copies
       are predicted by a model that never saw them. With one take per
       phrase, every fold trains on one phrase of the held-out speaker and
       tests on the other phrase, so this is a text-independent test.
    2. Logistic Regression, Random Forest and XGBoost are compared on these
       held-out predictions. The one with the lowest log loss is kept.
    3. Threshold, from held-out verification attempts:
         * genuine: a clip claiming its own speaker;
         * member impostor: a clip claiming one of the other 3 members
           (the "identity mismatch" case);
         * simulated stranger: leave-one-member-out. Each member is removed
           from training, and their clips claim each of the other 3 members.
       The threshold minimises  false-reject rate + 2 x false-accept rate
       (false accepts averaged over the two impostor types): approving a
       payment for the wrong person is worse than asking a member to repeat
       the phrase.
    4. Non-member .wav clips in ``data/unauthorized/voices/`` and
       ``Audio_features/data/audio/unauthorized/`` are scored with the final
       model, and their rejection rate is reported.
    5. The kept model is refit on all member clips and saved for the CLI.

Run from the project root:
    python src/train_voice_model.py                               # train, evaluate, save
    python src/train_voice_model.py verify <clip.wav> [member]    # verify one clip
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from src import config
except ModuleNotFoundError:
    import config

# Parfait's audio pipeline lives in Audio_features/ (not a package).
if str(config.AUDIO_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(config.AUDIO_PIPELINE_DIR))
import audio_pipeline as ap  # noqa: E402


def candidates() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=config.RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=config.RANDOM_STATE, verbosity=0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(config.AUDIO_FEATURES_CSV)
    df = df[df["member"].isin(config.MEMBERS)].reset_index(drop=True)
    return df, ap.feature_columns(df)


def unauthorized_clips() -> list[Path]:
    """Non-member .wav clips from data/unauthorized/voices/ and Parfait's folder."""
    clips = []
    for folder in [config.UNAUTHORIZED_VOICES_DIR, ap.UNAUTH_DIR]:
        if folder.exists():
            clips += ap.list_audio(folder)
    return clips


def fit_predict(model, X_tr, y_tr, X_te, n_classes: int) -> np.ndarray:
    """Fit on integer labels and return probabilities over all n_classes."""
    present, y_local = np.unique(y_tr, return_inverse=True)   # XGBoost needs 0..k-1
    fitted = clone(model).fit(X_tr, y_local)
    proba = np.zeros((len(X_te), n_classes))
    proba[:, present] = fitted.predict_proba(X_te)
    return proba


def scores(y: np.ndarray, proba: np.ndarray) -> dict:
    labels = np.arange(proba.shape[1])
    p = np.clip(proba, 1e-6, 1)
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", labels=labels),
        "log_loss": log_loss(y, p / p.sum(axis=1, keepdims=True), labels=labels),
    }


def claim_score(proba: np.ndarray, claimed: np.ndarray) -> np.ndarray:
    """P(claimed member) if the classifier's top pick is that member, else 0."""
    rows = np.arange(len(proba))
    return np.where(proba.argmax(axis=1) == claimed, proba[rows, claimed], 0.0)


FALSE_ACCEPT_COST = 2.0   # approving a payment for the wrong person costs 2x a retry


def verification_cost(t: float, genuine, member_imp, stranger) -> float:
    """False-reject rate + FALSE_ACCEPT_COST x mean false-accept rate."""
    far = ((member_imp >= t).mean() + (stranger >= t).mean()) / 2
    return float((genuine < t).mean() + FALSE_ACCEPT_COST * far)


def choose_threshold(genuine: np.ndarray, member_imp: np.ndarray, stranger: np.ndarray) -> float:
    """Threshold with the lowest verification cost (middle of the best range)."""
    cuts = np.unique(np.r_[genuine, member_imp, stranger])
    mids = np.r_[(cuts[:-1] + cuts[1:]) / 2, cuts[-1] + 1e-3]
    mids = mids[mids > 0]                       # score 0 = wrong member: always rejected
    cost = np.array([verification_cost(t, genuine, member_imp, stranger) for t in mids])
    best = np.flatnonzero(np.isclose(cost, cost.min()))
    return float(mids[best[len(best) // 2]])


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(verbose: bool = True) -> dict:
    df, cols = load_data()
    classes = np.array(sorted(df["member"].unique()))
    k = len(classes)
    y = np.searchsorted(classes, df["member"].to_numpy())
    groups = df["source_file"].to_numpy()
    X = df[cols].to_numpy()
    logo = list(LeaveOneGroupOut().split(X, y, groups))
    if verbose:
        print(f"{len(df)} rows from {df['source_file'].nunique()} recordings, {len(cols)} features, "
              f"{k} members, {len(logo)} folds")
    if df.groupby("member")["source_file"].nunique().min() < 2:
        raise ValueError("Every member needs at least 2 recordings for leave-one-recording-out CV.")

    # 1-2. Algorithm comparison on held-out predictions
    oof, rows = {}, []
    for name, model in candidates().items():
        proba = np.zeros((len(y), k))
        for tr, te in logo:
            proba[te] = fit_predict(model, X[tr], y[tr], X[te], k)
        oof[name] = proba
        rows.append({"model": name, **scores(y, proba)})
    comparison = pd.DataFrame(rows).set_index("model").sort_values(["log_loss", "accuracy"],
                                                                   ascending=[True, False])
    chosen = comparison.index[0]
    proba = oof[chosen]
    if verbose:
        print(comparison.round(4).to_string())
        print(f"Chosen: {chosen}")

    # 3. Verification attempts
    genuine = claim_score(proba, y)
    member_imp = np.concatenate([claim_score(proba, np.full(len(y), c))[y != c] for c in range(k)])
    stranger = []
    for out in range(k):
        inside = y != out
        p = fit_predict(candidates()[chosen], X[inside], y[inside], X[~inside], k)
        stranger.extend(np.concatenate([claim_score(p, np.full(len(p), c)) for c in range(k) if c != out]))
    stranger = np.array(stranger)
    threshold = round(choose_threshold(genuine, member_imp, stranger), 3)

    pred = proba.argmax(axis=1)
    per_aug = (pd.DataFrame({"augmentation": df["augmentation"], "correct": pred == y,
                             "accepted": genuine >= threshold})
               .groupby("augmentation", sort=False)
               .agg(accuracy=("correct", "mean"), accept_rate=("accepted", "mean")))
    per_member = (pd.DataFrame({"member": classes[y], "correct": pred == y,
                                "accepted": genuine >= threshold})
                  .groupby("member").agg(accuracy=("correct", "mean"),
                                         accept_rate=("accepted", "mean")))

    final = clone(candidates()[chosen]).fit(X, y)

    # 4. Real unauthorized voices, if any
    unauthorized = None
    clips = unauthorized_clips()
    if clips:
        U = np.array([[ap.extract_features(ap.load_clip(c))[f] for f in cols] for c in clips])
        u_proba = final.predict_proba(U)
        # Worst case: the stranger is paired with the face of whichever member
        # they sound most like, so they are accepted if ANY member passes.
        u_accept = u_proba.max(axis=1) >= threshold
        unauthorized = {
            "clips": len(clips),
            "reject_rate": round(float(1 - u_accept.mean()), 4),
            "per_clip": {
                str(c.relative_to(config.ROOT)): {
                    "closest_member": str(classes[int(p.argmax())]),
                    "confidence": round(float(p.max()), 3),
                    "rejected": bool(not a),
                }
                for c, p, a in zip(clips, u_proba, u_accept)
            },
        }

    result = {
        "df": df, "cols": cols, "y": y, "classes": classes, "comparison": comparison,
        "chosen": chosen, "oof": oof, "oof_proba": proba, "threshold": threshold,
        "genuine": genuine, "member_impostor": member_imp, "stranger": stranger,
        "genuine_accept_rate": float((genuine >= threshold).mean()),
        "member_impostor_reject_rate": float((member_imp < threshold).mean()),
        "stranger_reject_rate": float((stranger < threshold).mean()),
        "per_augmentation": per_aug, "per_member": per_member,
        "unauthorized": unauthorized, "final_model": final,
    }
    if verbose:
        print(f"Threshold P(claimed member) = {threshold}: genuine accepted "
              f"{result['genuine_accept_rate']:.1%}, other members rejected "
              f"{result['member_impostor_reject_rate']:.1%}, simulated strangers rejected "
              f"{result['stranger_reject_rate']:.1%}")
        print("Unauthorized voices:", unauthorized or "none yet (unauthorized folder is empty)")
    return result


def save(result: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result["final_model"],
        "feature_columns": result["cols"],
        "classes": list(result["classes"]),
        "threshold": result["threshold"],
    }
    joblib.dump(bundle, config.VOICE_MODEL_PATH)

    m = scores(result["y"], result["oof_proba"])
    cmp_ = result["comparison"]
    df = result["df"]
    meta = {
        "model": "voice_verification",
        "owner": config.MEMBER_NAMES["parfait"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": result["chosen"],
        "feature_columns": result["cols"],
        "classes": bundle["classes"],
        "threshold": result["threshold"],
        "threshold_metric": "P(claimed member | clip); accept only if the top prediction is "
                            "the claimed member and its probability >= threshold",
        "threshold_selection": f"minimises false-reject rate + {FALSE_ACCEPT_COST:g} x false-accept "
                               "rate on held-out verification attempts",
        "training_rows": int(len(df)),
        "recordings": int(df["source_file"].nunique()),
        "selection": "algorithm with the lowest log loss on leave-one-recording-out predictions",
        "metrics": {
            "held_out_test": {
                "evaluation": "leave-one-recording-out: each recording and its augmented "
                              "copies predicted by a model that never saw them (trained on "
                              "the speaker's other phrase only)",
                "rows": int(len(df)),
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
            "per_member": {
                mem: {k: round(float(v), 4) for k, v in r.items()}
                for mem, r in result["per_member"].iterrows()
            },
            "rejection": {
                "genuine_accept_rate": round(result["genuine_accept_rate"], 4),
                "member_impostor_reject_rate": round(result["member_impostor_reject_rate"], 4),
                "simulated_stranger_reject_rate": round(result["stranger_reject_rate"], 4),
                "simulated_stranger_method": "leave-one-member-out: each member removed from "
                                             "training; their clips claim each other member",
                "unauthorized_voices": result["unauthorized"] or "pending: no .wav samples in "
                                                                 "data/unauthorized/voices/ yet",
            },
        },
        "confusion_matrix": confusion_matrix(result["y"], result["oof_proba"].argmax(axis=1),
                                             labels=np.arange(len(result["classes"]))).tolist(),
    }
    config.VOICE_META_PATH.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Inference (used by the CLI app)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_model() -> dict:
    return joblib.load(config.VOICE_MODEL_PATH)


def verify_voice(audio: str | Path, expected_member: str | None = None) -> dict:
    """Does this clip belong to ``expected_member`` (the face result)?

    Without ``expected_member`` it just checks that the voice is confidently
    one of the enrolled members.
    """
    b = load_model()
    feats = ap.extract_features(ap.load_clip(audio))
    x = np.array([[feats[c] for c in b["feature_columns"]]])
    proba = b["model"].predict_proba(x)[0]
    classes = b["classes"]
    member = classes[int(proba.argmax())]
    conf = float(proba.max())

    if conf < b["threshold"]:
        reason = f"voice not recognised (confidence {conf:.2f} < {b['threshold']:.2f})"
    elif expected_member and member != expected_member.lower():
        reason = f"identity mismatch: voice sounds like '{member}', face was '{expected_member}'"
    else:
        reason = "voice verified"
    return {
        "approved": reason == "voice verified",
        "predicted_member": member,
        "confidence": round(conf, 3),
        "threshold": b["threshold"],
        "reason": reason,
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        r = verify_voice(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        print("APPROVED" if r["approved"] else "ACCESS DENIED", "-", r["reason"])
        print(json.dumps(r, indent=2))
    else:
        res = train_and_evaluate()
        save(res)
        print(f"\nSaved {config.VOICE_MODEL_PATH.relative_to(config.ROOT)} "
              f"and {config.VOICE_META_PATH.relative_to(config.ROOT)}")
