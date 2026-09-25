# Fake Review / Bot-Account Detector

Flags likely fake or bot-generated product reviews using **behavioral and
linguistic fingerprints** — burstiness, duplicate/near-duplicate phrasing,
rating-text mismatch, and reviewer behavior patterns — instead of just
sentiment analysis.

## Why this project

Most student review-analysis projects stop at sentiment classification
(positive/negative). This project instead tackles a harder, more realistic
problem: telling **authentic reviews apart from fake/bot-generated ones**,
similar to trust-and-safety work at companies like Amazon, Flipkart, and Yelp.

## Project structure

```
fake_review_detector/
├── app.py                        # Streamlit demo app
├── requirements.txt
├── data/
│   └── reviews.csv               # sample/synthetic dataset (generated)
├── models/
│   ├── fake_review_model.joblib  # trained model (generated)
│   └── metadata.json             # training metadata (generated)
└── src/
    ├── generate_sample_data.py   # creates a synthetic labeled dataset
    ├── feature_engineering.py    # core feature engineering logic
    ├── train_model.py            # trains + evaluates the classifier
    └── detect.py                 # CLI to score a CSV of reviews
```

## Quick start

```bash
# 1. create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. generate the sample dataset (or replace data/reviews.csv with your own)
python src/generate_sample_data.py

# 4. train the model
python src/train_model.py

# 5. run the interactive demo
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Scoring your own reviews from the command line

```bash
python src/detect.py --input data/your_reviews.csv --output data/scored.csv
```

Input CSV must contain these columns:

| column                        | meaning                                      |
|--------------------------------|-----------------------------------------------|
| `product_id`                   | groups reviews by product                     |
| `reviewer_name`                 | reviewer display name                          |
| `rating`                        | 1–5 star rating                                |
| `review_text`                   | the review text                                |
| `timestamp`                     | when the review was posted                     |
| `account_age_days`              | how old the reviewer account is                |
| `reviewer_total_reviews`        | how many reviews that reviewer has posted total|
| `reviewer_brand_concentration`  | 0–1, how concentrated their reviews are on one brand/seller |

## Using real scraped data instead of the sample dataset

`src/generate_sample_data.py` creates **synthetic** data so the whole
pipeline runs out of the box for demo purposes. For a real resume project,
replace `data/reviews.csv` with data you collect yourself, e.g.:

- A public labeled dataset such as the Ott et al. deceptive-opinion-spam
  corpus, or the Amazon Reviews dataset on Kaggle (fastest way to get
  ground-truth labels for evaluation).
- Your own scraper against public product review pages (respect each
  site's `robots.txt` and rate limits). `account_age_days`,
  `reviewer_total_reviews`, and `reviewer_brand_concentration` typically
  require visiting the reviewer's public profile page.

If you don't have ground-truth fake/genuine labels for your own scraped
data, switch `train_model.py` to an **unsupervised** approach instead
(e.g. `sklearn.ensemble.IsolationForest` on the same `FEATURE_COLUMNS`)
to flag anomalous reviews without needing labels — arguably a more
original and defensible approach for a resume project, since you're not
just fitting to someone else's labels.

## Methodology (feature engineering, the heart of the project)

| Feature | What it captures |
|---|---|
| `burst_count` | how many other reviews of the same product landed within a 60-minute window — a bot-attack signature |
| `max_similarity_same_product` | TF-IDF cosine similarity to the most similar other review of the same product — catches template reuse |
| `exclamation_density`, `superlative_count/ratio` | over-the-top, generic marketing-speak language common in fake reviews |
| `mismatch_score` | flags 5-star ratings paired with vague, short, generic text |
| `account_age_days`, `reviewer_total_reviews`, `reviewer_brand_concentration` | reviewer-level behavioral signals (new accounts, high volume, single-brand focus) |

## Model

Gradient-boosted classifier (XGBoost, falls back to scikit-learn's
RandomForest if `xgboost` isn't installed) trained on the engineered
features above. `train_model.py` prints a classification report,
confusion matrix, ROC-AUC, and feature importances, and saves the
trained model to `models/fake_review_model.joblib`.

## Possible extensions (good "future work" talking points)

- Swap in a transformer-based text embedding (e.g. sentence-transformers)
  in place of/alongside TF-IDF similarity for more robust duplicate
  detection against paraphrased fake reviews.
- Add SHAP values to explain individual predictions in the Streamlit app.
- Move from a single 60-minute burst window to a proper time-series
  burst-detection method (e.g. Kleinberg's burst detection algorithm).
- Deploy the Streamlit app publicly (Streamlit Community Cloud) so it's
  a live, clickable link on your resume.
