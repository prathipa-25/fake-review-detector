"""
feature_engineering.py
------------------------
Turns raw review rows into the engineered features that actually
carry signal for fake/bot review detection:

1. Burstiness       - how many reviews landed in a tight time window
                       for the same product
2. Duplicate/near-duplicate phrasing - text similarity vs other
                       reviews of the same product
3. Text statistics   - length, punctuation/exclamation density,
                       superlative word count, generic-language score
4. Rating-text mismatch - does an extremely positive rating pair
                       with vague/generic text
5. Reviewer behavior - account age, review count, brand concentration
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPERLATIVES = {
    "best", "amazing", "perfect", "excellent", "awesome", "incredible",
    "fantastic", "wonderful", "flawless", "outstanding", "superb"
}


def _exclamation_density(text: str) -> float:
    if not text:
        return 0.0
    return text.count("!") / max(len(text), 1)


def _superlative_count(text: str) -> int:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for w in words if w in SUPERLATIVES)


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z']+", text))


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_length"] = df["review_text"].str.len().fillna(0)
    df["word_count"] = df["review_text"].apply(_word_count)
    df["exclamation_density"] = df["review_text"].apply(_exclamation_density)
    df["superlative_count"] = df["review_text"].apply(_superlative_count)
    df["superlative_ratio"] = df["superlative_count"] / df["word_count"].replace(0, 1)
    return df


def add_rating_text_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags reviews that are 5-star but generically worded (short,
    high superlative ratio, low specific detail) - a common pattern
    in fake reviews, as opposed to genuine 5-star reviews which
    still tend to mention specifics.
    """
    df = df.copy()
    df["is_top_rating"] = (df["rating"] >= 5).astype(int)
    df["mismatch_score"] = (
        df["is_top_rating"] * df["superlative_ratio"] * (1 / (df["word_count"] + 1))
    )
    return df


def add_burstiness(df: pd.DataFrame, window_minutes: int = 60) -> pd.DataFrame:
    """
    For each review, counts how many other reviews of the SAME product
    were posted within `window_minutes` of it. High burstiness = many
    reviews landed in a very tight time cluster (classic bot-attack
    signature).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    burst_counts = []

    for product_id, group in df.groupby("product_id"):
        times = group["timestamp"].values.astype("datetime64[m]")
        times_sorted = np.sort(times)
        for t in group["timestamp"].values.astype("datetime64[m]"):
            window = np.timedelta64(window_minutes, "m")
            count = np.sum(np.abs(times_sorted - t) <= window) - 1  # exclude self
            burst_counts.append(count)

    # burst_counts was built in groupby order, need to realign to original df order
    df = df.sort_values("product_id").reset_index(drop=True)
    df["burst_count"] = burst_counts
    return df


def add_duplicate_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each product, computes each review's max cosine similarity
    (TF-IDF) to any OTHER review of the same product. Near-identical
    template reviews (common with paid/bot reviews) score high here.
    """
    df = df.copy()
    max_sims = np.zeros(len(df))

    for product_id, group in df.groupby("product_id"):
        idx = group.index.to_numpy()
        texts = group["review_text"].fillna("").tolist()
        if len(texts) < 2:
            continue
        vec = TfidfVectorizer(stop_words="english").fit_transform(texts)
        sim_matrix = cosine_similarity(vec)
        np.fill_diagonal(sim_matrix, 0)  # ignore self-similarity
        group_max = sim_matrix.max(axis=1)
        max_sims[idx] = group_max

    df["max_similarity_same_product"] = max_sims
    return df


FEATURE_COLUMNS = [
    "text_length",
    "word_count",
    "exclamation_density",
    "superlative_count",
    "superlative_ratio",
    "mismatch_score",
    "burst_count",
    "max_similarity_same_product",
    "account_age_days",
    "reviewer_total_reviews",
    "reviewer_brand_concentration",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full feature pipeline and returns a df ready for modeling."""
    df = add_text_stats(df)
    df = add_rating_text_mismatch(df)
    df = add_burstiness(df)
    df = add_duplicate_similarity(df)
    return df
