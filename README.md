# Project Name: Rental price forcasting
test_change
other_branch_change

**Author:** Group 6    
**Course:** MLOps: Master in Business Analytics and Data Sciense  
**Status:** Session 1 (Initialization)

---

## 1. Business Objective


* **The Goal:** Accurately predict the rent price of rental properties in the city of Madrid to determine if a listing is overpriced, underpriced or fairly priced.

* **The User:** Tenants who are seaching for rental properties in Madrid.

---

## 2. Success Metrics

* **Business KPI (The "Why"):**
  > Increase conversion rate by 5% in one year.

* **Technical Metric (The "How"):**
  > Model MAPE (Mean Absolute Percentage Error) < 15% on the test set.

* **Acceptance Criteria:**
  > It outperforms the baseline of the district median price model.

---

## 3. The Data

* **Source:** CSC with retal prices, location and structural charactersitics of the properties.
* **Target Variable:** Rental price
* **Sensitive Info:** None
  > *⚠️ **WARNING:** If the dataset contains sensitive data, it must NEVER be committed to GitHub. Ensure `data/` is in your `.gitignore`.*

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                # This file (Project definition)
├── environment.yml          # Dependencies (Conda/Pip)
├── config.yaml              # Global configuration (paths, params)
├── .env                     # Secrets placeholder
│
├── notebooks/               # Experimental sandbox
│   └── yourbaseline.ipynb   # From previous work
│
├── src/                     # Production code (The "Factory")
│   ├── __init__.py          # Python package
│   ├── load_data.py         # Ingest raw data
│   ├── clean_data.py        # Preprocessing & cleaning
│   ├── features.py          # Feature engineering
│   ├── validate.py          # Data quality checks
│   ├── train.py             # Model training & saving
│   ├── evaluate.py          # Metrics & plotting
│   ├── infer.py             # Inference logic
│   └── main.py              # Pipeline orchestrator
│
├── data/                    # Local storage (IGNORED by Git)
│   ├── raw/                 # Immutable input data
│   └── processed/           # Cleaned data ready for training
│
├── models/                  # Serialized artifacts (IGNORED by Git)
│
├── reports/                 # Generated metrics, plots, and figures
│
└── tests/                   # Automated tests
```

## 5. Execution Model

The full machine learning pipeline will eventually be executable through:

`python -m src.main`



