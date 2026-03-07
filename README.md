# Customer Segmentation & Clustering

Unsupervised machine learning project to segment customers based on purchasing behavior and demographics using the marketing campaign dataset.

---

## Dataset

**File:** `data/raw/marketing_campaign.csv`  
**Shape:** 2,240 rows × 29 columns (tab-separated)

| Category | Columns |
|---|---|
| Demographics | `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income` |
| Household | `Kidhome`, `Teenhome` |
| Spending | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchases | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Campaigns | `AcceptedCmp1–5`, `Response` |
| Other | `Dt_Customer`, `Recency`, `Complain`, `Z_CostContact`, `Z_Revenue` |

---

## Project Structure

```
customer-segmentation-clustering/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned dataset
├── notebooks/
│   └── project_analysis.ipynb  # Main EDA & modeling notebook
├── outputs/                  # Saved plots / results
└── README.md
```

---

## Approach

### 1. Exploratory Data Analysis (EDA)

- **Age** — Derived from `Year_Birth`; outliers (age > 100) removed.
- **Education** — Grouped into 3 tiers: *Undergraduate*, *Graduate*, *Postgraduate*.
- **Marital Status** — `Alone` merged into `Single`; rare/invalid entries (`Absurd`, `YOLO`) dropped.
- **Income** — Extreme outlier (> 600 000) removed; noted right-skewed distribution requiring scaling.
- **Children** — `Kidhome` + `Teenhome` combined into `Total_Children`.
- **Campaigns** — Five campaign acceptance flags aggregated into `Total_Campaign_Accepted`.
- **Recency** — Distribution of days since last purchase visualised.
- **Complaints** — Minority class (~20 / 2,188 customers).
- **Zero-variance columns** — `Z_CostContact` and `Z_Revenue` dropped (constant values).
- **Membership date** — `Dt_Customer` converted to `Customer_Tenure_Days`.

### 2. Feature Engineering

| New Feature | Description |
|---|---|
| `Age` | `2025 − Year_Birth` |
| `Education_Group` | Collapsed education into 3 groups |
| `Total_Children` | `Kidhome + Teenhome` |
| `Total_Campaign_Accepted` | Sum of 5 campaign flags |
| `Customer_Tenure_Days` | Days between join date and latest date in dataset |

### 3. Clustering

- **Algorithm:** K-Means  
- **Why:** Distance-based — requires scaled features and outlier removal.
- **`Response`** column intentionally kept out of clustering features (used for post-cluster analysis).

---

## Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing & K-Means clustering |

---

## How to Run

1. Place `marketing_campaign.csv` (tab-separated) in `data/raw/`.
2. Open and run `notebooks/project_analysis.ipynb` top-to-bottom.
3. Cleaned data is optionally exported to `data/processed/marketing_data_cleaned.csv`.