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
        "name": "🔵 High Value Loyalists",
        "description": "Highest income, maximum spending, campaign-responsive, prefer catalogue shopping.",
        "recommendation": "Target with premium products, exclusive catalogues, and personalized campaigns."
    },
    2: {
        "name": "🟡 Middle Class Actives",
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
    mnt_fish            = st.number_input("Spent on Fish ($)",          0, 300, 10)
    mnt_sweet           = st.number_input("Spent on Sweets ($)",        0, 300, 10)
    mnt_gold            = st.number_input("Spent on Gold ($)",          0, 400, 50)

with col3:
    num_web_purchases   = st.number_input("Web Purchases",              0, 30, 5)
    num_store_purchases = st.number_input("Store Purchases",            0, 20, 5)
    num_catalog         = st.number_input("Catalog Purchases",          0, 30, 2)
    num_deals           = st.number_input("Deal Purchases",             0, 15, 1)
    num_web_visits      = st.number_input("Web Visits/Month",           0, 20, 5)
    campaigns_accepted  = st.number_input("Campaigns Accepted (0-6)",  0, 6, 0)

    

if st.button("🔍 Predict My Segment"):

    total_spending        = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    total_purchases       = num_web_purchases + num_store_purchases + num_catalog + num_deals
    spending_per_purchase = total_spending / (total_purchases + 1)

    mnt_wines_log  = np.log1p(mnt_wines)
    mnt_fruits_log = np.log1p(mnt_fruits)
    mnt_meat_log   = np.log1p(mnt_meat)
    mnt_fish_log   = np.log1p(mnt_fish)
    mnt_sweet_log  = np.log1p(mnt_sweet)
    mnt_gold_log   = np.log1p(mnt_gold)

    input_data = pd.DataFrame([[
        income, recency, mnt_wines_log, mnt_fruits_log, mnt_meat_log,
        mnt_fish_log, mnt_sweet_log, mnt_gold_log,
        num_deals, num_web_purchases, num_catalog, num_store_purchases,
        num_web_visits, age, total_children, campaigns_accepted,
        tenure_days, total_purchases, total_spending, spending_per_purchase,
        1, 0, 0, 0, 0
    ]], columns=[
        'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
        'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 'Total_Children',
        'Total_Campaign_Accepted', 'Customer_Tenure_Days', 'Total_Purchases',
        'Total_Spending', 'Spending_Per_Purchase',
        'Marital_Status_Partnered', 'Marital_Status_Single',
        'Marital_Status_Widow', 'Education_Group_Postgraduate',
        'Education_Group_Undergraduate'
    ])

    input_scaled    = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)
    input_pca       = pca.transform(input_scaled_df)
    cluster         = model.predict(input_pca)[0]

    info = cluster_info[cluster]
    st.success(f"### Predicted Segment: {info['name']}")
    st.info(f"**Profile:** {info['description']}")
    st.warning(f"**Recommendation:** {info['recommendation']}")
    st.markdown("---")
