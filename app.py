import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Load saved models ──────────────────────────
scaler = joblib.load('outputs/models/scaler.pkl')
pca = joblib.load('outputs/models/pca.pkl')
model = joblib.load('outputs/models/kmeans_model.pkl')

# ── Cluster info ───────────────────────────────
cluster_info = {
    0: {
        "name": "🟣 Budget Conscious Families",
        "description": "Low income, high number of children, price-sensitive customers who browse frequently but spend minimally.",
        "recommendation": "Target with discount deals, family bundles, and budget-friendly promotions."
    },
    1: {
        "name": "🟡 High Value Loyalists",
        "description": "Highest income, maximum spending, campaign-responsive, prefer catalogue shopping.",
        "recommendation": "Target with premium products, exclusive catalogues, and personalized campaigns."
    },
    2: {
        "name": "🔵 Middle Class Actives",
        "description": "Mid-range income, loyal customers with moderate spending across all channels.",
        "recommendation": "Reward loyalty with membership programs and mid-range product promotions."
    }
}

# ── App UI ─────────────────────────────────────
st.title("🛍️ Customer Segment Predictor")
st.markdown("Enter customer details below to predict which segment they belong to.")

# ── Input fields ───────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    income              = st.number_input("Annual Income ($)",          1000, 200000, 50000)
    recency             = st.number_input("Days Since Last Purchase",   0, 100, 30)
    total_children      = st.number_input("Number of Children",         0, 5, 1)
    age                 = st.number_input("Age",                        18, 100, 40)
    tenure_days         = st.number_input("Customer Tenure (Days)",     0, 1000, 300)

with col2:
    mnt_wines           = st.number_input("Spent on Wines ($)",         0, 1500, 100)
    mnt_fruits          = st.number_input("Spent on Fruits ($)",        0, 200, 20)
    mnt_meat            = st.number_input("Spent on Meat ($)",          0, 1500, 100)
    mnt_gold            = st.number_input("Spent on Gold ($)",          0, 400, 50)

with col3:
    num_web_purchases   = st.number_input("Web Purchases",              0, 30, 5)
    num_store_purchases = st.number_input("Store Purchases",            0, 20, 5)
    num_catalog         = st.number_input("Catalog Purchases",          0, 30, 2)
    num_web_visits      = st.number_input("Web Visits/Month",           0, 20, 5)
    campaigns_accepted  = st.number_input("Campaigns Accepted (0-6)",  0, 6, 0)
    
# ── Derived features ───────────────────────────
total_spending        = mnt_wines + mnt_fruits + mnt_meat + mnt_gold
total_purchases       = num_web_purchases + num_store_purchases + num_catalog
spending_per_purchase = total_spending / (total_purchases + 1)

if st.button("🔍 Predict My Segment"):

    input_data = pd.DataFrame([[
    income, age, recency, total_spending, total_purchases,
    spending_per_purchase, tenure_days, total_children, campaigns_accepted,
    mnt_wines, mnt_meat, mnt_fruits, mnt_gold,
    num_web_purchases, num_store_purchases, num_catalog,
    num_web_visits, 1, 0
]], columns=[
    'Income', 'Age', 'Recency', 'Total_Spending', 'Total_Purchases',
    'Spending_Per_Purchase', 'Customer_Tenure_Days', 'Total_Children',
    'Total_Campaign_Accepted', 'MntWines', 'MntMeatProducts', 'MntFruits',
    'MntGoldProds', 'NumWebPurchases', 'NumStorePurchases',
    'NumCatalogPurchases', 'NumWebVisitsMonth', 'Marital_Status_Partnered',
    'Education_Group_Postgraduate'
])



    input_scaled = scaler.transform(input_data) 
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)
    input_pca    = pca.transform(input_scaled_df)     
    cluster      = model.predict(input_pca)[0]

    info = cluster_info[cluster]
    st.success(f"### Predicted Segment: {info['name']}")
    st.info(f"**Profile:** {info['description']}")
    st.warning(f"**Recommendation:** {info['recommendation']}")
    st.markdown("---")

    st.markdown(f"**Cluster ID:** `{cluster}`")
