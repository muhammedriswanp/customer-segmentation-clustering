# resave_models.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# Load processed data
df = pd.read_csv('data/processed/marketing_data_cleaned.csv')

# Encode categorical columns
df = pd.get_dummies(df, columns=['Marital_Status', 'Education_Group'], dtype=int)

feature_cols = [
    'Income', 'Age', 'Recency', 'Total_Spending', 'Total_Purchases',
    'Spending_Per_Purchase', 'Customer_Tenure_Days', 'Total_Children',
    'Total_Campaign_Accepted', 'MntWines', 'MntMeatProducts', 'MntFruits',
    'MntGoldProds', 'NumWebPurchases', 'NumStorePurchases',
    'NumCatalogPurchases', 'NumWebVisitsMonth', 'Marital_Status_Partnered',
    'Education_Group_Postgraduate'
]

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)

# PCA
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X_scaled)

# KMeans
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X_pca)

# Save
joblib.dump(scaler, 'outputs/models/scaler.pkl')
joblib.dump(pca,    'outputs/models/pca.pkl')
joblib.dump(model,  'outputs/models/kmeans_model.pkl')

import sklearn
print("✅ All models resaved!")
print("sklearn version:", sklearn.__version__)