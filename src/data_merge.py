"""Data merge, cleaning and feature engineering for the product recommendation model.

Owner: Hassan Adelani

Pipeline:
    1. Load the two raw CSVs.
    2. Clean social profiles: drop exact duplicates, normalise strings,
       derive the integer join key from ``customer_id_new`` ("A178" -> 178).
    3. Clean transactions: fix dtypes, check IDs and value ranges, impute
       missing ``customer_rating``, flag ``purchase_amount`` outliers (IQR).
    4. Aggregate social profiles to ONE row per customer. Customers have
       several profiles, so joining raw rows would duplicate transactions.
    5. LEFT JOIN transactions -> aggregated social features on customer_id.
       Customers without a profile get imputed features and
       ``has_social_profile = 0``.
    6. Validate the merge (row count preserved, keys unique, no nulls).
    7. Engineer date features and leakage-safe customer-history features
       (computed from each customer's *earlier* transactions only).
    8. Save to ``data/processed/merged_customer_data.csv``.

Run from the project root:
    python src/data_merge.py
"""

from __future__ import annotations

import pandas as pd

try:  # imported as a package (notebooks, other modules)
    from src import config
except ModuleNotFoundError:  # run as a script: python src/data_merge.py
    import config

PLATFORMS = ["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
SENTIMENT_SCORE = {"Positive": 1, "Neutral": 0, "Negative": -1}
CATEGORIES = ["Books", "Clothing", "Electronics", "Groceries", "Sports"]

TARGET = "product_category"

# Placeholder categories. Avoid "None"/"NA", which pandas reads back as NaN.
NO_HISTORY = "no_history"   # cust_prev_category on a customer's first purchase
NO_PROFILE = "no_profile"   # dominant_platform for customers without a profile

# Columns describing the purchase itself (known only after it happens).
# They stay in the merged dataset for EDA. The product model should use
# the leakage-safe "cust_prev_*" history features instead.
POST_PURCHASE_COLUMNS = ["purchase_amount", "amount_bucket", "customer_rating", "rating_was_missing"]

# Identifier / bookkeeping columns that are never model features.
ID_COLUMNS = ["transaction_id", "customer_id", "purchase_date"]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (social_profiles, transactions) exactly as stored on disk."""
    social = pd.read_csv(config.SOCIAL_PROFILES_CSV)
    transactions = pd.read_csv(config.TRANSACTIONS_CSV)
    return social, transactions


# ---------------------------------------------------------------------------
# 2. Clean social profiles
# ---------------------------------------------------------------------------
def clean_social_profiles(social: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicates, normalise text, derive the integer ``customer_id`` key."""
    df = social.copy()
    report = {"rows_before": len(df)}

    for col in ["customer_id_new", "social_media_platform", "review_sentiment"]:
        df[col] = df[col].astype("string").str.strip()
    df["social_media_platform"] = df["social_media_platform"].str.title().replace(
        {"Linkedin": "LinkedIn", "Tiktok": "TikTok"}
    )
    df["review_sentiment"] = df["review_sentiment"].str.title()

    report["exact_duplicates_dropped"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    df["customer_id"] = df["customer_id_new"].str.lstrip("A").astype(int)
    df["engagement_score"] = df["engagement_score"].astype(int)
    df["purchase_interest_score"] = df["purchase_interest_score"].astype(float)

    # Domain checks: the merge must not silently absorb bad categories.
    unknown_platforms = set(df["social_media_platform"]) - set(PLATFORMS)
    unknown_sentiments = set(df["review_sentiment"]) - set(SENTIMENTS)
    assert not unknown_platforms, f"Unexpected platforms: {unknown_platforms}"
    assert not unknown_sentiments, f"Unexpected sentiments: {unknown_sentiments}"
    assert df["engagement_score"].between(0, 100).all()
    assert df["purchase_interest_score"].between(0, 5).all()

    report["rows_after"] = len(df)
    report["nulls"] = int(df.isna().sum().sum())
    report["unique_customers"] = int(df["customer_id"].nunique())
    return df, report


# ---------------------------------------------------------------------------
# 3. Clean transactions
# ---------------------------------------------------------------------------
def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean_transactions(transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix dtypes, check keys and ranges, impute ratings, flag amount outliers."""
    df = transactions.copy().rename(columns={"customer_id_legacy": "customer_id"})
    report = {"rows_before": len(df)}

    df["customer_id"] = df["customer_id"].astype(int)
    df["transaction_id"] = df["transaction_id"].astype(int)
    df["purchase_amount"] = df["purchase_amount"].astype(float)
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], format="%Y-%m-%d")
    df[TARGET] = df[TARGET].astype("string").str.strip().str.title()
    df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")

    report["exact_duplicates"] = int(df.duplicated().sum())
    report["duplicate_transaction_ids"] = int(df["transaction_id"].duplicated().sum())
    df = df.drop_duplicates(subset="transaction_id").reset_index(drop=True)

    unknown_categories = set(df[TARGET]) - set(CATEGORIES)
    assert not unknown_categories, f"Unexpected categories: {unknown_categories}"
    assert (df["purchase_amount"] > 0).all()
    assert df["customer_rating"].dropna().between(1, 5).all()

    # Missing ratings: a customer's own median is a better guess than the
    # global median because rating behaviour is personal. Fall back to the
    # global median for customers with no other rated purchase. The flag
    # records which values were imputed.
    report["missing_ratings"] = int(df["customer_rating"].isna().sum())
    df["rating_was_missing"] = df["customer_rating"].isna().astype(int)
    customer_median = df.groupby("customer_id")["customer_rating"].transform("median")
    global_median = df["customer_rating"].median()
    filled_by_customer = df["customer_rating"].isna() & customer_median.notna()
    report["ratings_imputed_customer_median"] = int(filled_by_customer.sum())
    report["ratings_imputed_global_median"] = report["missing_ratings"] - report["ratings_imputed_customer_median"]
    report["global_rating_median"] = float(global_median)
    df["customer_rating"] = df["customer_rating"].fillna(customer_median).fillna(global_median)

    # Outliers are flagged, not removed: amounts in the 51-495 range are
    # plausible purchases, and with 150 rows every observation matters.
    low, high = iqr_bounds(df["purchase_amount"])
    df["amount_is_outlier"] = (~df["purchase_amount"].between(low, high)).astype(int)
    report["amount_iqr_bounds"] = (float(low), float(high))
    report["amount_outliers"] = int(df["amount_is_outlier"].sum())

    report["rows_after"] = len(df)
    report["unique_customers"] = int(df["customer_id"].nunique())
    return df, report


# ---------------------------------------------------------------------------
# 4. Aggregate social profiles per customer
# ---------------------------------------------------------------------------
def aggregate_social_profiles(social: pd.DataFrame) -> pd.DataFrame:
    """Collapse a customer's profiles into one row of numeric features."""
    g = social.groupby("customer_id")

    agg = pd.DataFrame(
        {
            "n_social_profiles": g.size(),
            "n_platforms": g["social_media_platform"].nunique(),
            "avg_engagement": g["engagement_score"].mean(),
            "max_engagement": g["engagement_score"].max(),
            "avg_purchase_interest": g["purchase_interest_score"].mean(),
            "max_purchase_interest": g["purchase_interest_score"].max(),
            "sentiment_score": g["review_sentiment"].apply(
                lambda s: s.map(SENTIMENT_SCORE).mean()
            ),
        }
    )

    platform_counts = (
        pd.crosstab(social["customer_id"], social["social_media_platform"])
        .reindex(columns=PLATFORMS, fill_value=0)
        .add_prefix("plat_")
    )
    sentiment_share = (
        pd.crosstab(social["customer_id"], social["review_sentiment"], normalize="index")
        .reindex(columns=SENTIMENTS, fill_value=0.0)
        .rename(columns=lambda c: f"pct_{c.lower()}")
    )

    # Dominant platform = where the customer is most active (ties -> the
    # profile with the highest engagement).
    dominant = (
        social.sort_values(["customer_id", "engagement_score"], ascending=[True, False])
        .groupby("customer_id")["social_media_platform"]
        .agg(lambda s: s.value_counts(sort=True).index[0])
        .rename("dominant_platform")
    )

    out = agg.join([platform_counts, sentiment_share, dominant]).reset_index()
    return out


SOCIAL_FEATURES_ZERO_FILL = (
    ["n_social_profiles", "n_platforms", "sentiment_score"]
    + [f"plat_{p}" for p in PLATFORMS]
    + [f"pct_{s.lower()}" for s in SENTIMENTS]
)
SOCIAL_FEATURES_MEDIAN_FILL = [
    "avg_engagement",
    "max_engagement",
    "avg_purchase_interest",
    "max_purchase_interest",
]


# ---------------------------------------------------------------------------
# 5. Merge
# ---------------------------------------------------------------------------
def merge_datasets(transactions: pd.DataFrame, social_agg: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """LEFT JOIN transactions to aggregated social features, then impute gaps.

    Why a LEFT join on transactions:
      * The prediction unit is a transaction (target = product_category),
        so every labelled transaction should be kept.
      * An INNER join would drop all transactions from customers without a
        social profile (33 of 150 here). That costs 22% of an already
        tiny dataset.
      * ``social_agg`` has exactly one row per customer, so the join cannot
        duplicate transactions (validated with ``validate="many_to_one"``).
    """
    report = {
        "inner_join_rows": int(transactions.merge(social_agg, on="customer_id", how="inner").shape[0]),
        # Profiles with no transactions carry no label, so they cannot be
        # training rows. Their absence from the merged set is expected.
        "social_only_customers": sorted(
            set(social_agg["customer_id"]) - set(transactions["customer_id"])
        ),
    }

    merged = transactions.merge(
        social_agg, on="customer_id", how="left", validate="many_to_one", indicator=True
    )
    merged["has_social_profile"] = (merged["_merge"] == "both").astype(int)
    merged = merged.drop(columns="_merge")

    report["left_join_rows"] = len(merged)
    report["matched_rows"] = int(merged["has_social_profile"].sum())
    report["unmatched_rows"] = int((merged["has_social_profile"] == 0).sum())
    report["unmatched_customers"] = sorted(
        merged.loc[merged["has_social_profile"] == 0, "customer_id"].unique().tolist()
    )
    report["nulls_before_imputation"] = int(merged.isna().sum().sum())

    # No profile means no observed activity: counts and shares become 0 and
    # sentiment is neutral (0). Engagement/interest levels are unknown rather
    # than zero, so they get the median over customers who DO have profiles.
    # has_social_profile lets the model tell the two groups apart.
    merged[SOCIAL_FEATURES_ZERO_FILL] = merged[SOCIAL_FEATURES_ZERO_FILL].fillna(0)
    medians = social_agg[SOCIAL_FEATURES_MEDIAN_FILL].median()
    merged[SOCIAL_FEATURES_MEDIAN_FILL] = merged[SOCIAL_FEATURES_MEDIAN_FILL].fillna(medians)
    merged["dominant_platform"] = merged["dominant_platform"].fillna(NO_PROFILE)
    for col in ["n_social_profiles", "n_platforms"] + [f"plat_{p}" for p in PLATFORMS]:
        merged[col] = merged[col].astype(int)

    report["nulls_after_imputation"] = int(merged.isna().sum().sum())
    return merged, report


# ---------------------------------------------------------------------------
# 6. Validate
# ---------------------------------------------------------------------------
def validate_merge(
    merged: pd.DataFrame,
    transactions: pd.DataFrame,
    social: pd.DataFrame,
) -> pd.DataFrame:
    """Run post-merge checks; raise if any fail. Returns a check table."""
    checks = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check(
        "Row count preserved",
        len(merged) == len(transactions),
        f"{len(merged)} merged vs {len(transactions)} transactions",
    )
    check("transaction_id unique", merged["transaction_id"].is_unique)
    check(
        "Same transaction_id set",
        set(merged["transaction_id"]) == set(transactions["transaction_id"]),
    )
    check("No nulls after imputation", merged.isna().sum().sum() == 0, f"{int(merged.isna().sum().sum())} nulls")
    check("Target has no nulls", merged[TARGET].notna().all())

    # Every matched customer's aggregated features must equal a direct
    # recomputation from the cleaned profile rows.
    matched = merged[merged["has_social_profile"] == 1]
    mismatches = 0
    for cid, row in matched.drop_duplicates("customer_id").set_index("customer_id").iterrows():
        rows = social[social["customer_id"] == cid]
        mismatches += int(
            len(rows) != row["n_social_profiles"]
            or abs(rows["engagement_score"].mean() - row["avg_engagement"]) > 1e-9
            or abs(rows["purchase_interest_score"].mean() - row["avg_purchase_interest"]) > 1e-9
        )
    check("Aggregates match source rows", mismatches == 0, f"{mismatches} mismatching customers")

    unmatched_ids = set(merged.loc[merged["has_social_profile"] == 0, "customer_id"])
    check(
        "Unmatched customers truly lack profiles",
        unmatched_ids.isdisjoint(set(social["customer_id"])),
    )

    table = pd.DataFrame(checks)
    failed = table[~table["passed"]]
    if not failed.empty:
        raise AssertionError(f"Merge validation failed:\n{failed.to_string(index=False)}")
    return table


# ---------------------------------------------------------------------------
# 7. Feature engineering
# ---------------------------------------------------------------------------
def _prev_mode(categories: list[str]) -> str:
    """Most frequent earlier category; ties go to the most recent one."""
    if not categories:
        return NO_HISTORY
    counts = pd.Series(categories).value_counts()
    top = counts[counts == counts.max()].index
    for cat in reversed(categories):
        if cat in top:
            return cat
    return NO_HISTORY  # unreachable


def engineer_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Add date features and leakage-safe customer-history features.

    History features use ONLY the customer's transactions strictly before
    the current one (sorted by date). Each row's features therefore describe
    what was known *before* that purchase, as they would be at prediction
    time.
    """
    df = merged.sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)

    # Date features
    df["purchase_month"] = df["purchase_date"].dt.month
    df["purchase_dayofweek"] = df["purchase_date"].dt.day_of_week  # Mon=0
    df["is_weekend"] = (df["purchase_dayofweek"] >= 5).astype(int)

    # Spend band (tertiles). Describes this purchase, so it is post-purchase.
    df["amount_bucket"] = pd.qcut(df["purchase_amount"], q=3, labels=["low", "medium", "high"]).astype(str)

    # Customer history (prior transactions only)
    g = df.groupby("customer_id", sort=False)
    df["cust_prev_txn_count"] = g.cumcount()
    df["is_first_purchase"] = (df["cust_prev_txn_count"] == 0).astype(int)
    df["cust_prev_avg_amount"] = g["purchase_amount"].transform(lambda s: s.shift().expanding().mean())
    df["cust_prev_avg_rating"] = g["customer_rating"].transform(lambda s: s.shift().expanding().mean())
    df["days_since_prev_txn"] = g["purchase_date"].diff().dt.days

    prev_cat = {}
    for _, grp in g:
        history: list[str] = []
        for idx, cat in zip(grp.index, grp[TARGET]):
            prev_cat[idx] = _prev_mode(history)
            history.append(cat)
    df["cust_prev_category"] = pd.Series(prev_cat)

    # First purchases have no history. Use the global median for the averages
    # (is_first_purchase marks these rows) and -1 as a "no previous visit"
    # sentinel for the day gap.
    df["cust_prev_avg_amount"] = df["cust_prev_avg_amount"].fillna(df["purchase_amount"].median())
    df["cust_prev_avg_rating"] = df["cust_prev_avg_rating"].fillna(df["customer_rating"].median())
    df["days_since_prev_txn"] = df["days_since_prev_txn"].fillna(-1).astype(int)

    return df


COLUMN_ORDER = [
    # identifiers
    "transaction_id", "customer_id", "purchase_date",
    # target
    TARGET,
    # raw transaction attributes (post-purchase, see POST_PURCHASE_COLUMNS)
    "purchase_amount", "amount_bucket", "amount_is_outlier", "customer_rating", "rating_was_missing",
    # date features
    "purchase_month", "purchase_dayofweek", "is_weekend",
    # customer history (leakage-safe)
    "cust_prev_txn_count", "is_first_purchase", "cust_prev_avg_amount",
    "cust_prev_avg_rating", "days_since_prev_txn", "cust_prev_category",
    # social profile features
    "has_social_profile", "n_social_profiles", "n_platforms",
    "avg_engagement", "max_engagement", "avg_purchase_interest", "max_purchase_interest",
    "sentiment_score", "pct_positive", "pct_neutral", "pct_negative",
    *[f"plat_{p}" for p in PLATFORMS], "dominant_platform",
]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_merged_dataset(save: bool = True, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Run the full pipeline. Returns (final_df, reports)."""
    social_raw, tx_raw = load_raw()
    social, social_report = clean_social_profiles(social_raw)
    tx, tx_report = clean_transactions(tx_raw)
    social_agg = aggregate_social_profiles(social)
    merged, merge_report = merge_datasets(tx, social_agg)
    checks = validate_merge(merged, tx, social)
    final = engineer_features(merged)[COLUMN_ORDER]
    final = final.sort_values("transaction_id").reset_index(drop=True)

    assert final.isna().sum().sum() == 0, "Nulls introduced during feature engineering"
    assert len(final) == len(tx_raw)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        final.to_csv(config.MERGED_CSV, index=False, date_format="%Y-%m-%d")

    reports = {
        "social": social_report,
        "transactions": tx_report,
        "merge": merge_report,
        "validation": checks,
    }
    if verbose:
        _print_summary(final, reports, saved=save)
    return final, reports


def _print_summary(final: pd.DataFrame, reports: dict, saved: bool) -> None:
    s, t, m = reports["social"], reports["transactions"], reports["merge"]
    print("Social profiles : "
          f"{s['rows_before']} rows -> {s['rows_after']} after dropping "
          f"{s['exact_duplicates_dropped']} duplicates ({s['unique_customers']} customers)")
    print("Transactions    : "
          f"{t['rows_after']} rows, {t['unique_customers']} customers, "
          f"{t['missing_ratings']} missing ratings imputed "
          f"({t['ratings_imputed_customer_median']} customer median, "
          f"{t['ratings_imputed_global_median']} global median), "
          f"{t['amount_outliers']} amount outliers flagged")
    print("Merge           : "
          f"LEFT join {m['left_join_rows']} rows "
          f"({m['matched_rows']} matched / {m['unmatched_rows']} without profile); "
          f"INNER join would keep {m['inner_join_rows']}; "
          f"{len(m['social_only_customers'])} profile-only customers have no transactions (no label)")
    print("Validation      : "
          f"{int(reports['validation']['passed'].sum())}/{len(reports['validation'])} checks passed")
    print(f"Final dataset   : {final.shape[0]} rows x {final.shape[1]} columns, "
          f"{int(final.isna().sum().sum())} nulls")
    if saved:
        print(f"Saved to        : {config.MERGED_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    build_merged_dataset()
