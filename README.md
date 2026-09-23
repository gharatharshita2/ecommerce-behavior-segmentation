# E-Commerce Customer Behavior Analysis & Segmentation

An end-to-end data science pipeline that analyzes e-commerce session behavior, segments visitors into behavioral groups using K-Means clustering, and predicts purchase conversion using classification models — deployed as a live prediction API with automated testing via CI.

## Overview

This project answers two core business questions using real browsing session data:
1. **What kinds of visitors does the site get?** → Behavioral segmentation via K-Means
2. **Will a given session end in a purchase?** → Conversion prediction via Random Forest

## Tech Stack

- **Python (OOP)** — `DataLoader` and `FeatureEngineer` classes for reusable data pipelines
- **SQL / SQLite** — behavioral aggregation queries
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — K-Means clustering, Logistic Regression, Random Forest, GridSearchCV
- **Flask** — REST API for real-time predictions
- **Matplotlib / Seaborn** — visualizations
- **Pytest + GitHub Actions** — automated testing and CI

## Dataset

[Online Shoppers Purchasing Intention Dataset](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention) (UCI ML Repository via Kaggle) — 12,330 anonymized e-commerce sessions, 18 features covering page views, durations, bounce/exit rates, Google Analytics PageValues, and purchase outcome.

**Note:** this dataset is session-level, not customer-level — there is no customer ID, so segments reflect visit behavior rather than tracked individual customers across visits.

## Key Findings

- **Baseline conversion rate: 15.47%** of sessions result in a purchase.
- **New visitors convert nearly 2x more often than returning visitors** (24.9% vs 13.9%), despite returning visitors spending over double the time on product pages — suggesting returning visitors browse across multiple sessions before buying.
- **`PageValues` is by far the strongest single predictor of conversion** (correlation 0.49), far ahead of page counts or durations.
- **November has both the highest traffic and highest conversion rate** (25.4%), consistent with holiday shopping season; February sees minimal traffic and a 1.6% conversion rate.

## Customer Segments (K-Means, k=4)

| Segment | Sessions | Conversion Rate | Profile |
|---|---|---|---|
| Window Shoppers | 1,022 | 0.6% | Fast in-and-out, high bounce/exit rates, zero PageValues |
| Typical Browsers | 9,733 | 11.3% | Baseline majority behavior |
| Deep Researchers | 901 | 31.0% | Most pages/time browsed, moderate PageValues |
| High-Intent Shoppers | 674 | 78.2% | Small group, dominated by high PageValues, near-certain to convert |

Segments were validated against actual purchase outcomes (not used during clustering) — the near-monotonic increase in conversion rate across segments confirms they capture meaningful behavioral differences, not arbitrary groupings.

## Classification Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (balanced) | 85.8% | 0.533 | 0.691 | 0.602 |
| Random Forest | 87.8% | 0.589 | 0.699 | 0.640 |
| Random Forest (GridSearchCV tuned) | 87.7% | 0.584 | **0.709** | **0.641** |

Given the 15.47% class imbalance, models were optimized for **F1/recall** rather than raw accuracy — a naive "always predict no purchase" baseline already achieves ~84.5% accuracy without being useful. `class_weight='balanced'` and hyperparameter tuning were used to reduce missed conversions.

## Running the Project

```bash
# clone and set up environment
git clone https://github.com/gharatharshita2/ecommerce-behavior-segmentation.git
cd ecommerce-behavior-segmentation
pip install -r requirements.txt

# rebuild the database and run the pipeline
python load_data.py
python explore_sql.py
python eda.py
python clustering.py
python classification.py

# run the API
python app.py
```

## API Usage

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Administrative": 3, "Administrative_Duration": 120.5, "Informational": 1, "Informational_Duration": 30.0, "ProductRelated": 45, "ProductRelated_Duration": 1800.0, "BounceRates": 0.01, "ExitRates": 0.02, "PageValues": 25.5, "TotalPages": 49, "TotalDuration": 1950.5, "ProductFocusRatio": 0.92}'
```

Response:
```json
{
  "will_purchase": true,
  "purchase_probability": 0.64
}
```

## Testing & CI

Automated tests (`pytest`) validate the data pipeline's core assumptions. GitHub Actions runs these on every push — see the badge/Actions tab for current status.

## Limitations & Future Work

- Session-level data means true per-customer RFM analysis (recency/frequency/monetary tied to an individual) isn't possible with this dataset — a real deployment would use customer-linked data.
- `BounceRates` and `ExitRates` are highly correlated (0.91) — feature selection or dimensionality reduction could simplify the model.
- Hyperparameter tuning yielded marginal gains over defaults, suggesting further improvement would need better features rather than different model settings.