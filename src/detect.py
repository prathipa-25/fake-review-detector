"""
detect.py
------------------------
Loads the trained model and scores review data for fake/bot
likelihood.

Usage:
    # score a whole csv of reviews (must have the same raw columns
    # as data/reviews.csv: product_id, reviewer_name, rating,
    # review_text, timestamp, account_age_days,
    # reviewer_total_reviews, reviewer_brand_concentration)
    python src/detect.py --input data/reviews.csv --output data/scored.csv
"""

import argparse
import joblib
import pandas as pd

from feature_engineering import build_features, FEATURE_COLUMNS


def score_dataframe(df: pd.DataFrame, model_path="models/fake_review_model.joblib") -> pd.DataFrame:
    model = joblib.load(model_path)
    feats = build_features(df)
    X = feats[FEATURE_COLUMNS]
    feats["fake_probability"] = model.predict_proba(X)[:, 1]
    feats["predicted_label"] = (feats["fake_probability"] >= 0.5).map(
        {True: "FAKE", False: "GENUINE"}
    )
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV of reviews")
    parser.add_argument("--output", required=True, help="Path to write scored CSV")
    parser.add_argument("--model", default="models/fake_review_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    scored = score_dataframe(df, args.model)

    cols_to_show = [
        "review_id", "product_id", "reviewer_name", "rating",
        "review_text", "fake_probability", "predicted_label"
    ]
    cols_to_show = [c for c in cols_to_show if c in scored.columns]
    scored[cols_to_show].to_csv(args.output, index=False)

    print(f"Scored {len(scored)} reviews -> {args.output}")
    print(scored["predicted_label"].value_counts())


if __name__ == "__main__":
    main()
