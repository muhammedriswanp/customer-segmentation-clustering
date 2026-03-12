# Customer Segmentation & Clustering 

Groups customers into distinct segments based on purchasing behavior and demographics using unsupervised machine learning.

## Dataset

**Source:** Marketing Campaign dataset (`data/raw/marketing_campaign.csv`)  
**Size:** 2,240 rows × 29 columns (tab-separated)  
**Features:** Demographics, spending amounts, purchase channels, and campaign responses

## Approach

- **EDA** — outlier removal (age > 100, income > 600K), education grouping, marital status cleanup
- **Preprocessing** — duplicates removed, missing income dropped, irrelevant columns (`ID`, `Response`, `Z_*`, `Complain`) dropped (`preprocessing.py`)
- **Feature Engineering** — derived `Age`, `Total_Children`, `Total_Campaign_Accepted`, `Customer_Tenure_Days`, `Total_Spending`, `Total_Purchases`, `Spending_Per_Purchase`; log-transform on skewed spend cols; one-hot encoding (`feature_engineering.py`)
- **Dimensionality Reduction** — PCA retaining 90% variance (`clustering.py`)
- **Clustering** — optimal K via Elbow + Silhouette scores; final model K-Means (K = 3); Agglomerative used for comparison (`clustering.py`)
- **Evaluation** — Davies–Bouldin Index for cluster quality (`evaluation.py`)

## Models Used

| Model | Type |
|---|---|
| K-Means | Partition-based ✅ Best |
| Agglomerative | Hierarchical (comparison) |

## Results

**Best Model:** K-Means with **3 clusters**

| Cluster | Label | Profile |
|---|---|---|
| 0 | 🟣 Budget Conscious Families | Low income, many children, price-sensitive, frequent web visits |
| 1 | 🟡 High Value Loyalists | Highest income, max spending, campaign-responsive, prefer catalogues |
| 2 | 🔵 Middle Class Actives | Mid-range income, loyal, moderate spending across all channels |

Top features: `Income`, `Total_Spending`, `Total_Purchases`, `Age`, `Total_Campaign_Accepted`

## Project Structure

```
customer-segmentation-clustering/
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Cleaned & clustered output CSV
├── notebooks/
│   └── project_analysis.ipynb       # EDA & exploration notebook
├── outputs/
│   ├── models/                      # scaler.pkl, pca.pkl, kmeans_model.pkl
│   └── clusters/                    # Cluster assignment CSV
│   └── reports/                     # Contains all the reports
├── src/
│   ├── preprocessing.py             # Cleaning & outlier removal
│   ├── feature_engineering.py       # Feature creation, encoding & scaling
│   ├── clustering.py                # PCA, KMeans, Agglomerative helpers
│   ├── evaluation.py                # Davies–Bouldin scoring
│   └── pipeline.py                  # End-to-end pipeline — trains & saves models
│
├── app.py                           # Streamlit customer segment predictor
├── main.py                          # main file to run the pipeline
└── requirements.txt
```

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models and save artefacts (scaler, PCA, KMeans)
python main.py

# 3. Run the Streamlit app
python -m streamlit run app.py
```
