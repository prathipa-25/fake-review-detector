"""
app.py
------------------------
Streamlit demo for the Fake Review / Bot Detector project.

Run:
    streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

from feature_engineering import build_features, FEATURE_COLUMNS

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="wide")

MODEL_PATH = "models/fake_review_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def model_available():
    return os.path.exists(MODEL_PATH)


st.title("🕵️ Fake Review / Bot-Account Detector")
st.caption(
    "Flags likely fake or bot-generated reviews using behavioral and "
    "linguistic fingerprints (burstiness, duplicate phrasing, rating-text "
    "mismatch) — not just sentiment."
)

if not model_available():
    st.error(
        "No trained model found. Run these first:\n\n"
        "```\npython src/generate_sample_data.py\npython src/train_model.py\n```"
    )
    st.stop()

model = load_model()

tab1, tab2 = st.tabs(["🔍 Score a single review", "📄 Score a CSV of reviews"])

with tab1:
    st.subheader("Try a single review")
    col1, col2 = st.columns(2)
    with col1:
        review_text = st.text_area(
            "Review text",
            "Best product ever!!! Amazing quality, super fast delivery, highly recommend!!!",
            height=120,
        )
        rating = st.slider("Rating", 1, 5, 5)
    with col2:
        account_age_days = st.number_input("Reviewer account age (days)", 0, 5000, 5)
        reviewer_total_reviews = st.number_input("Reviewer's total review count", 0, 500, 30)
        reviewer_brand_concentration = st.slider(
            "Reviewer's brand concentration (0=diverse, 1=only this brand)", 0.0, 1.0, 0.9
        )

    if st.button("Analyze review", type="primary"):
        single_df = pd.DataFrame([{
            "review_id": 1,
            "product_id": "DEMO",
            "reviewer_name": "demo_user",
            "rating": rating,
            "review_text": review_text,
            "timestamp": datetime.now(),
            "account_age_days": account_age_days,
            "reviewer_total_reviews": reviewer_total_reviews,
            "reviewer_brand_concentration": reviewer_brand_concentration,
        }])

        feats = build_features(single_df)
        X = feats[FEATURE_COLUMNS]
        prob = model.predict_proba(X)[0, 1]

        st.metric("Fake probability", f"{prob*100:.1f}%")
        if prob >= 0.5:
            st.error("⚠️ Likely FAKE / bot-generated review")
        else:
            st.success("✅ Likely GENUINE review")

        with st.expander("See engineered features used for this prediction"):
            st.dataframe(feats[FEATURE_COLUMNS].T.rename(columns={0: "value"}))

with tab2:
    st.subheader("Batch-score a CSV of reviews")
    st.caption(
        "CSV must have columns: product_id, reviewer_name, rating, review_text, "
        "timestamp, account_age_days, reviewer_total_reviews, reviewer_brand_concentration"
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "review_id" not in df.columns:
            df["review_id"] = range(1, len(df) + 1)

        feats = build_features(df)
        X = feats[FEATURE_COLUMNS]
        feats["fake_probability"] = model.predict_proba(X)[:, 1]
        feats["predicted_label"] = (feats["fake_probability"] >= 0.5).map(
            {True: "FAKE", False: "GENUINE"}
        )

        st.write(f"Scored {len(feats)} reviews.")
        st.bar_chart(feats["predicted_label"].value_counts())

        show_cols = ["review_id", "product_id", "reviewer_name", "rating",
                     "review_text", "fake_probability", "predicted_label"]
        show_cols = [c for c in show_cols if c in feats.columns]
        st.dataframe(
            feats[show_cols].sort_values("fake_probability", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            "Download scored CSV",
            feats[show_cols].to_csv(index=False),
            file_name="scored_reviews.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Model: gradient-boosted classifier trained on engineered features "
    "(burstiness, duplicate phrasing, rating-text mismatch, reviewer behavior). "
    "Built with scikit-learn / XGBoost + Streamlit."
)
