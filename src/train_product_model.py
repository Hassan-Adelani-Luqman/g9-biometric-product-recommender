"""Product recommendation model (Phase 3A).

Owner: Hassan Adelani Luqman

Predicts which product category a customer will buy next, from the merged
dataset built by ``data_merge.py``. Only features known *before* a purchase
are used (date, customer history, social profile), so the model can run for
a customer who is about to buy and has not bought yet.

Evaluation protocol (150 rows, 5 near-balanced classes):
    1. Stratified 80/20 split -> 120 training rows, 30 held-out test rows.
    2. On the training rows, for 3 algorithms x 3 feature sets: a small
       hyperparameter grid (5-fold stratified CV, scored on log loss), then
       repeated stratified 5-fold CV (5 x 10) of the tuned model for stable
       estimates.
    3. The combination with the lowest mean CV log loss is kept. It must
       beat the class-prior baseline on both log loss and accuracy.
    4. The held-out test set is scored once with the kept model.
    5. Customer-grouped CV (no customer in both train and test) shows how
       much of the performance comes from having seen the customer before.
    6. The kept model is refit on all 150 rows and saved for the CLI app.

Run from the project root:
    python src/train_product_model.py               # train, evaluate, save
    python src/train_product_model.py predict 187   # predict for one customer
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

try:  # imported as a package (notebooks, the CLI app)
    from src import config
    from src import data_merge as dm
except ModuleNotFoundError:  # run as a script: python src/train_product_model.py
    import config
    import data_merge as dm

RANDOM_STATE = config.RANDOM_STATE
CLASSES = dm.CATEGORIES  # alphabetical, same order as LabelEncoder

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["cust_prev_category", "dominant_platform"]
NUMERIC_FEATURES = [
    # purchase date (known at prediction time: it is "today")
    "purchase_month", "purchase_dayofweek", "is_weekend",
    # customer history, computed from earlier transactions only
    "cust_prev_txn_count", "is_first_purchase", "cust_prev_avg_amount",
    "cust_prev_avg_rating", "days_since_prev_txn",
    # social profile
    "has_social_profile", "n_social_profiles", "avg_engagement", "avg_purchase_interest",
    "sentiment_score", "pct_neutral", "pct_negative",
    *[f"plat_{p}" for p in dm.PLATFORMS],
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Feature sets compared during model selection, as (numeric, categorical).
# With 150 rows and 5 classes, 22 features let a model memorise noise. The
# EDA found only cust_prev_category (chi-squared p = 0.015) and, weakly,
# purchase_month (p = 0.055) related to the target, so the two smaller sets
# test whether dropping the rest generalises better. The choice is made by
# CV on the training split only, like any other hyperparameter.
FEATURE_SETS = {
    "all features": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    "previous category + month": ([], ["cust_prev_category", "purchase_month"]),
    "previous category": ([], ["cust_prev_category"]),
}

# Left out on purpose:
#   * post-purchase columns (amount, amount_bucket, rating, rating flag and
#     amount_is_outlier) describe the purchase being predicted -> leakage;
#   * one column of each redundant pair found in the EDA (|r| >= 0.85):
#     n_platforms (~ n_social_profiles), max_engagement (~ avg_engagement),
#     max_purchase_interest (~ avg_purchase_interest), pct_positive
#     (~ sentiment_score).
EXCLUDED_FEATURES = {
    "post_purchase": [*dm.POST_PURCHASE_COLUMNS, "amount_is_outlier"],
    "redundant": ["n_platforms", "max_engagement", "max_purchase_interest", "pct_positive"],
    "identifiers": dm.ID_COLUMNS,
}


def preprocessor(feature_set: str = "all features") -> ColumnTransformer:
    """Scale numeric features, one-hot encode categorical ones. Columns not
    in the feature set are dropped, so every pipeline takes the full row."""
    numeric, categorical = FEATURE_SETS[feature_set]
    steps = []
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(steps, verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# Candidate algorithms (each with a deliberately small grid: 120 rows)
# ---------------------------------------------------------------------------
def candidates() -> dict[str, tuple[object, dict]]:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.03, 0.1, 0.3, 1.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
            ),
            {"clf__max_depth": [3, None], "clf__min_samples_leaf": [1, 5]},
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob", n_estimators=200, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
                n_jobs=1, verbosity=0,
            ),
            {"clf__max_depth": [2, 3], "clf__min_child_weight": [1, 3]},
        ),
    }


def make_pipeline(clf, feature_set: str = "all features") -> Pipeline:
    return Pipeline([("prep", preprocessor(feature_set)), ("clf", clf)])


BASELINE = "Baseline (class prior)"


def model_name(algorithm: str, feature_set: str) -> str:
    return f"{algorithm} | {feature_set}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, LabelEncoder]:
    """Return (merged df, X, encoded y, label encoder)."""
    df = pd.read_csv(config.MERGED_CSV, parse_dates=["purchase_date"])
    encoder = LabelEncoder().fit(CLASSES)
    y = encoder.transform(df[dm.TARGET])
    return df, df[FEATURES], y, encoder


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
LABELS = np.arange(len(CLASSES))


def score(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Accuracy, macro F1, log loss and top-3 accuracy from probabilities."""
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro", labels=LABELS, zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=LABELS),
        "top3_accuracy": top_k_accuracy_score(y_true, proba, k=3, labels=LABELS),
    }


def cross_validate(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, splitter, groups=None) -> pd.DataFrame:
    """One row of metrics per fold. A custom loop (instead of sklearn's
    cross_validate) so log loss and top-3 work even when a fold misses a class."""
    def one_fold(train_idx, test_idx):
        model = clone(pipe).fit(X.iloc[train_idx], y[train_idx])
        return score(y[test_idx], _full_proba(model, X.iloc[test_idx]))

    rows = Parallel(n_jobs=-1)(delayed(one_fold)(tr, te) for tr, te in splitter.split(X, y, groups))
    return pd.DataFrame(rows)


def _full_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """predict_proba with one column per class, even if a class was absent
    from the training fold."""
    proba = model.predict_proba(X)
    if proba.shape[1] == len(CLASSES):
        return proba
    full = np.zeros((len(X), len(CLASSES)))
    full[:, model.classes_] = proba
    return full


def summarise(folds: pd.DataFrame) -> dict[str, float]:
    out = {}
    for col in folds.columns:
        out[f"{col}_mean"] = float(folds[col].mean())
        out[f"{col}_std"] = float(folds[col].std())
    return out


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(verbose: bool = True) -> dict:
    """Run the full protocol (see module docstring). Returns everything the
    notebook needs: fitted models, CV folds, test predictions, chosen name."""
    df, X, y, encoder = load_data()
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = X.iloc[idx_train], X.iloc[idx_test], y[idx_train], y[idx_test]

    tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    eval_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)

    baseline = BASELINE
    tuned: dict[str, Pipeline] = {baseline: make_pipeline(DummyClassifier(strategy="prior"))}
    best_params: dict[str, dict] = {baseline: {}}
    spec: dict[str, tuple[str, str]] = {baseline: ("Baseline (class prior)", "none")}
    for feature_set in FEATURE_SETS:
        for algorithm, (clf, grid) in candidates().items():
            name = model_name(algorithm, feature_set)
            search = GridSearchCV(
                make_pipeline(clf, feature_set), grid, cv=tune_cv, scoring="neg_log_loss", n_jobs=-1
            )
            search.fit(X_train, y_train)
            tuned[name] = search.best_estimator_
            best_params[name] = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
            spec[name] = (algorithm, feature_set)

    cv_folds = {name: cross_validate(pipe, X_train, y_train, eval_cv) for name, pipe in tuned.items()}
    cv_summary = pd.DataFrame({name: summarise(f) for name, f in cv_folds.items()}).T
    cv_summary.insert(0, "feature_set", [spec[n][1] for n in cv_summary.index])
    cv_summary.insert(0, "algorithm", [spec[n][0] for n in cv_summary.index])

    # Select on log loss: the same criterion as the tuning, a proper scoring
    # rule (rewards honest probabilities, which the CLI displays), and far
    # less noisy than accuracy/F1 on 24-row validation folds.
    models = [n for n in tuned if n != baseline]
    chosen = min(models, key=lambda n: cv_summary.loc[n, "log_loss_mean"])
    beats_baseline = (
        cv_summary.loc[chosen, "log_loss_mean"] < cv_summary.loc[baseline, "log_loss_mean"]
        and cv_summary.loc[chosen, "accuracy_mean"] > cv_summary.loc[baseline, "accuracy_mean"]
    )
    if not beats_baseline:
        raise RuntimeError("No model beats the class-prior baseline; do not ship it.")

    # Held-out test: each tuned model, fit on the 120 training rows, scored once.
    fitted = {name: clone(pipe).fit(X_train, y_train) for name, pipe in tuned.items()}
    test_proba = {name: _full_proba(m, X_test) for name, m in fitted.items()}
    test_summary = pd.DataFrame({name: score(y_test, p) for name, p in test_proba.items()}).T

    # Sensitivity check: customers never shared between train and test folds,
    # i.e. how well each model does for a customer it has never seen.
    group_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grouped = {
        name: summarise(cross_validate(pipe, X, y, group_cv, groups=df["customer_id"]))
        for name, pipe in tuned.items()
    }

    # Final model: the chosen configuration refit on all 150 rows.
    final_model = clone(tuned[chosen]).fit(X, y)

    result = {
        "df": df, "X": X, "y": y, "encoder": encoder,
        "idx_train": idx_train, "idx_test": idx_test,
        "tuned": tuned, "best_params": best_params, "spec": spec, "fitted_on_train": fitted,
        "cv_folds": cv_folds, "cv_summary": cv_summary,
        "test_proba": test_proba, "test_summary": test_summary,
        "grouped_cv": pd.DataFrame(grouped).T,
        "chosen": chosen, "baseline": baseline, "final_model": final_model,
    }
    if verbose:
        _print_report(result)
    return result


def save(result: dict) -> None:
    """Save the final model and its metadata to models/."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["final_model"], config.PRODUCT_MODEL_PATH)

    chosen, baseline = result["chosen"], result["baseline"]
    cv, test = result["cv_summary"], result["test_summary"]
    df = result["df"]
    y_test = result["y"][result["idx_test"]]
    meta = {
        "model": "product_recommendation",
        "owner": "Hassan Adelani Luqman",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": result["spec"][chosen][0],
        "feature_set": result["spec"][chosen][1],
        "params": result["best_params"][chosen],
        "classes": CLASSES,
        "numeric_features": FEATURE_SETS[result["spec"][chosen][1]][0],
        "categorical_features": FEATURE_SETS[result["spec"][chosen][1]][1],
        "input_columns": FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "training_rows": len(df),
        "data_as_of": str(df["purchase_date"].max().date()),
        "selection": "algorithm x feature set with the lowest mean log loss in repeated "
                     "stratified 5-fold CV (5 x 10) on the 120-row training split",
        "metrics": {
            "cv_5x10_train_split": {
                k: round(float(cv.loc[chosen, f"{k}_mean"]), 4)
                for k in ["accuracy", "f1_macro", "log_loss", "top3_accuracy"]
            },
            "held_out_test": {
                "rows": int(len(y_test)),
                **{k: round(float(v), 4) for k, v in test.loc[chosen].items()},
            },
            "customer_grouped_cv": {
                k: round(float(result["grouped_cv"].loc[chosen, f"{k}_mean"]), 4)
                for k in ["accuracy", "f1_macro", "log_loss", "top3_accuracy"]
            },
            "baseline_class_prior": {
                "cv_accuracy": round(float(cv.loc[baseline, "accuracy_mean"]), 4),
                "cv_log_loss": round(float(cv.loc[baseline, "log_loss_mean"]), 4),
                "test_accuracy": round(float(test.loc[baseline, "accuracy"]), 4),
            },
        },
        "test_confusion_matrix": confusion_matrix(
            y_test, result["test_proba"][chosen].argmax(axis=1), labels=LABELS
        ).tolist(),
    }
    config.PRODUCT_META_PATH.write_text(json.dumps(meta, indent=2) + "\n")


def _print_report(r: dict) -> None:
    cols = ["accuracy_mean", "f1_macro_mean", "log_loss_mean", "top3_accuracy_mean"]
    pd.set_option("display.width", 120)
    print("Repeated stratified 5-fold CV (5 x 10) on the 120 training rows:")
    print(r["cv_summary"][cols].round(3).sort_values("log_loss_mean").to_string())
    print("\nHeld-out test set (30 rows):")
    print(r["test_summary"].round(3).sort_values("log_loss").to_string())
    print("\nCustomer-grouped 5-fold CV (all 150 rows, no customer in both train and test):")
    print(r["grouped_cv"][cols].round(3).sort_values("log_loss_mean").to_string())
    print(f"\nChosen: {r['chosen']} {r['best_params'][r['chosen']]}")


# ---------------------------------------------------------------------------
# Inference (used by the CLI app)
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, object] = {}


def load_model() -> tuple[Pipeline, dict]:
    if "model" not in _MODEL_CACHE:
        _MODEL_CACHE["model"] = joblib.load(config.PRODUCT_MODEL_PATH)
        _MODEL_CACHE["meta"] = json.loads(config.PRODUCT_META_PATH.read_text())
    return _MODEL_CACHE["model"], _MODEL_CACHE["meta"]


def customer_feature_row(customer_id: int, as_of: str | pd.Timestamp | None = None,
                         df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the feature row for this customer's *next* purchase.

    History features summarise ALL of the customer's recorded transactions,
    the same way ``data_merge.engineer_features`` summarises the earlier ones
    for each training row. ``as_of`` is the purchase date to predict for. It
    defaults to the day after the last date in the dataset, so the demo is
    reproducible and the day gap stays in the range the model was trained on.
    """
    if df is None:
        df = pd.read_csv(config.MERGED_CSV, parse_dates=["purchase_date"])
    rows = df[df["customer_id"] == int(customer_id)].sort_values(["purchase_date", "transaction_id"])
    if rows.empty:
        raise KeyError(f"customer_id {customer_id} is not in the merged dataset")

    as_of = (df["purchase_date"].max() + pd.Timedelta(days=1)) if as_of is None else pd.Timestamp(as_of)
    last = rows.iloc[-1]

    feat = {c: last[c] for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES}  # social profile columns
    feat.update(
        purchase_month=as_of.month,
        purchase_dayofweek=as_of.day_of_week,
        is_weekend=int(as_of.day_of_week >= 5),
        cust_prev_txn_count=len(rows),
        is_first_purchase=0,
        cust_prev_avg_amount=float(rows["purchase_amount"].mean()),
        cust_prev_avg_rating=float(rows["customer_rating"].mean()),
        days_since_prev_txn=int((as_of - last["purchase_date"]).days),
        cust_prev_category=dm._prev_mode(rows[dm.TARGET].tolist()),
    )
    return pd.DataFrame([feat])[FEATURES]


def predict_for_customer(customer_id: int, as_of: str | None = None) -> tuple[str, dict[str, float]]:
    """Return (predicted category, {category: probability}), highest first."""
    model, meta = load_model()
    proba = model.predict_proba(customer_feature_row(customer_id, as_of))[0]
    ranked = sorted(zip(meta["classes"], proba.tolist()), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], dict(ranked)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "predict":
        category, probs = predict_for_customer(int(sys.argv[2]))
        print(f"Customer {sys.argv[2]} -> {category}")
        for cat, p in probs.items():
            print(f"  {cat:<12}{p:6.1%}")
    else:
        res = train_and_evaluate()
        save(res)
        print(f"\nSaved {config.PRODUCT_MODEL_PATH.relative_to(config.ROOT)} "
              f"and {config.PRODUCT_META_PATH.relative_to(config.ROOT)}")
