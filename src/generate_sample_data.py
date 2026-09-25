"""
generate_sample_data.py
------------------------
Creates a synthetic e-commerce review dataset with a ground-truth
'is_fake' label, so the whole pipeline can be trained and demoed
without needing to scrape a live site first.

Once you're ready to use REAL data, replace data/reviews.csv with
your own scraped/collected reviews (see README for the scraper
template) — the rest of the pipeline does not need to change, as
long as the columns match.

Run:
    python src/generate_sample_data.py
"""

import random
import string
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

GENUINE_TEMPLATES = [
    "The {item} works well but the {part} feels a bit cheap after a few weeks.",
    "Decent {item} for the price. Delivery took longer than expected though.",
    "I've been using this {item} for a month now, mixed feelings about the {part}.",
    "Good value overall. The {part} could be better but I'm satisfied.",
    "Not bad, does what it says. {part} quality is average.",
    "Works fine for basic use, wouldn't recommend for heavy use though.",
    "Packaging was damaged but the {item} itself was fine.",
    "Took a while to get used to it, but now I like the {item}.",
]

FAKE_TEMPLATES = [
    "Best {item} ever!!! Amazing quality, super fast delivery, highly recommend to everyone!!!",
    "Excellent product, excellent seller, excellent service, five stars all the way!!!",
    "Amazing {item}, changed my life, will buy again, best purchase ever!!!",
    "This {item} is absolutely perfect in every single way, no complaints at all!!!",
    "Wow just wow, perfect {item}, perfect packaging, perfect everything, 5 stars!!!",
    "Highly recommend this amazing {item}, super value for money, must buy now!!!",
]

ITEMS = ["phone case", "bluetooth speaker", "backpack", "watch", "charger",
         "headphones", "shoes", "blender", "lamp", "keyboard"]
PARTS = ["build quality", "battery life", "material", "stitching", "sound quality",
         "screen", "strap", "switches", "handle"]


def random_username(fake=False):
    if fake:
        # bot-like usernames: word + long digit string
        prefixes = ["user", "shopper", "buyer", "customer", "deal"]
        return f"{random.choice(prefixes)}{random.randint(100000, 999999)}"
    else:
        first = random.choice(["Arun", "Priya", "Karthik", "Sneha", "Rahul",
                                "Divya", "Vignesh", "Meera", "Suresh", "Anitha"])
        return f"{first}{random.randint(1, 99)}"


def make_review(is_fake, product_id, base_date):
    item = random.choice(ITEMS)
    part = random.choice(PARTS)
    if is_fake:
        text = random.choice(FAKE_TEMPLATES).format(item=item, part=part)
        rating = 5
        # bots often post in tight clusters
        ts = base_date + timedelta(minutes=random.randint(0, 20))
        account_age_days = random.randint(0, 10)
        reviewer_total_reviews = random.randint(15, 60)
        reviewer_brand_concentration = round(random.uniform(0.85, 1.0), 2)
    else:
        text = random.choice(GENUINE_TEMPLATES).format(item=item, part=part)
        rating = random.choice([2, 3, 3, 4, 4, 4, 5, 5])
        ts = base_date + timedelta(days=random.randint(0, 120),
                                    hours=random.randint(0, 23))
        account_age_days = random.randint(60, 2000)
        reviewer_total_reviews = random.randint(1, 25)
        reviewer_brand_concentration = round(random.uniform(0.05, 0.5), 2)

    return {
        "review_id": None,  # filled later
        "product_id": product_id,
        "reviewer_name": random_username(is_fake),
        "rating": rating,
        "review_text": text,
        "timestamp": ts,
        "account_age_days": account_age_days,
        "reviewer_total_reviews": reviewer_total_reviews,
        "reviewer_brand_concentration": reviewer_brand_concentration,
        "is_fake": int(is_fake),
    }


def generate_dataset(n_products=25, genuine_per_product=12, fake_burst_products_ratio=0.4):
    rows = []
    review_id = 1
    for p in range(n_products):
        product_id = f"P{p:03d}"
        base_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 300))

        # genuine reviews, spread out over time
        for _ in range(genuine_per_product):
            row = make_review(False, product_id, base_date)
            row["review_id"] = review_id
            review_id += 1
            rows.append(row)

        # some products get a "fake review burst" (bot attack)
        if random.random() < fake_burst_products_ratio:
            burst_date = base_date + timedelta(days=random.randint(0, 300))
            n_fake = random.randint(8, 20)
            for _ in range(n_fake):
                row = make_review(True, product_id, burst_date)
                row["review_id"] = review_id
                review_id += 1
                rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/reviews.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} reviews -> {out_path}")
    print(df["is_fake"].value_counts())
