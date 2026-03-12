import pandas as pd
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features, scale_features
from src.clustering import apply_pca, run_kmeans
import joblib

FEATURE_COLS = [
    'Income', 'Age', 'Recency', 'Total_Spending', 'Total_Purchases',
    'Spending_Per_Purchase', 'Customer_Tenure_Days', 'Total_Children',
    'Total_Campaign_Accepted', 'MntWines', 'MntMeatProducts', 'MntFruits',
    'MntGoldProds', 'NumWebPurchases', 'NumStorePurchases',
    'NumCatalogPurchases', 'NumWebVisitsMonth', 'Marital_Status_Partnered',
    'Education_Group_Postgraduate'
]

def run_pipeline(input_path, output_path):
    print("Loading Data...")
    df = pd.read_csv(input_path, sep='\t')

    print("Preprocessing...")
    df = preprocess(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Scaling features ...")
    X_scaled, scaler = scale_features(df, FEATURE_COLS)

    print("Applying PCA...")
    X_pca, pca = apply_pca(X_scaled)

    print("Clustering...")
    model, labels = run_kmeans(X_pca, n_clusters=3)
    df['KMeans_Cluster'] = labels

    print("Saving models...")
    joblib.dump(scaler, 'outputs/models/scaler.pkl')
    joblib.dump(pca, "outputs/models/pca.pkl")
    joblib.dump(model, "outputs/models/kmeans_model.pkl")

    print("Saving results...")
    df.to_csv(output_path, index=False)

    print("Pipeline complete!")
    return df