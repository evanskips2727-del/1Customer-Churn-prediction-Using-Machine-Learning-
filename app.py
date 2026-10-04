import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up page configurations
st.set_page_config(page_title="NexAfrica Churn Analytics Dashboard", layout="wide")

# App Header
st.title("📊 Interactive Customer Churn Dashboard")
st.markdown("### NexAfrica ML Internship Project")

# Safely load the processed dataset
DATA_PATH = "data/processed_data/processed_data.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Ensure 'Churn' column is present and structured cleanly for plotting
        if 'Churn' in df.columns:
            # Map numeric 1/0 to Yes/No if it was encoded during processing
            df['Churn_Label'] = df['Churn'].map({1: 'Yes', 0: 'No'}).fillna(df['Churn'])
        else:
            df['Churn_Label'] = 'Unknown'
        return df
    else:
        return None

df = load_data()

if df is None:
    st.error(f"Could not find the dataset at `{DATA_PATH}`. Please verify your file path repository configuration.")
else:
    # Sidebar Navigation
    st.sidebar.header("Navigation Menu")
    page = st.sidebar.radio("Go to:", [
        "Data Overview", 
        "Demographics & Services", 
        "Charges & Financials", 
        "Feature Importance"
    ])

    # 1. Data Overview Page
    if page == "Data Overview":
        st.header("🏢 Dataset Exploration")
        st.write("Below is a sample of your cleaned and processed dataset from Kaggle's Telco Customer Churn:")
        
        # Interactive DataFrame widget
        st.dataframe(df.head(100), use_container_width=True)
        
        # Quick metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers Displayed", len(df))
        if 'Churn' in df.columns:
            churn_rate = (df['Churn'].astype(str).str.lower().isin(['yes', '1']).sum() / len(df)) * 100
            col2.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
        col3.metric("Total Metrics Tracked", len(df.columns) - 1)

    # 2. Demographics & Services Page
    elif page == "Demographics & Services":
        st.header("👥 Demographics & Service Profiles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Senior Citizen Distribution")
            if 'SeniorCitizen' in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='SeniorCitizen', hue='Churn_Label', palette='pastel', ax=ax)
                ax.set_title("Churn Breakdown for Senior Citizens (1) vs Non-Seniors (0)")
                st.pyplot(fig)
            else:
                st.info("Column 'SeniorCitizen' not detected in processed data.")
                
        with col2:
            st.subheader("Internet Service Providers")
            if 'InternetService' in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='InternetService', hue='Churn_Label', palette='muted', ax=ax)
                ax.set_title("Churn Count by Internet Service Provider")
                st.pyplot(fig)
            else:
                # Fallback for fiber optic indicator flag if already encoded
                fiber_col = [c for c in df.columns if 'fiber' in c.lower()]
                if fiber_col:
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, x=fiber_col[0], hue='Churn_Label', palette='muted', ax=ax)
                    ax.set_title(f"Churn Count by {fiber_col[0]}")
                    st.pyplot(fig)
                else:
                    st.info("Internet service identifiers not found.")

    # 3. Charges & Financials Page
    elif page == "Charges & Financials":
        st.header("💳 Financial Charges & Account Types Analysis")
        
        # Recreating the box plot for Monthly Charges Range vs Churn
        if 'MonthlyCharges' in df.columns:
            st.subheader("Monthly Charges Ranges vs Churn Status")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x='Churn_Label', y='MonthlyCharges', palette='Set2', ax=ax)
            st.pyplot(fig)
            
        # Recreating the Scatter Plot: Total Charges vs Monthly Charges
        if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
            st.subheader("Total Charges vs Monthly Charges Distribution")
            
            # Interactive Filter Slider for Monthly Charges range
            min_val, max_val = float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max())
            charge_filter = st.slider("Filter Monthly Charges Range:", min_val, max_val, (min_val, max_val))
            
            filtered_df = df[(df['MonthlyCharges'] >= charge_filter[0]) & (df['MonthlyCharges'] <= charge_filter[1])]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=filtered_df, x='MonthlyCharges', y='TotalCharges', hue='Churn_Label', alpha=0.6, ax=ax)
            ax.set_title("Visualization 7: Total Charges vs Monthly Charges Scatter Plot")
            st.pyplot(fig)

    # 4. Feature Importance Page
    elif page == "Feature Importance":
        st.header("🤖 Machine Learning Insights")
        st.write("Here are the top feature rules driving your deployment's underlying XGBoost algorithm calculations:")
        
        # Displaying your pre-computed static feature importance plot
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", caption="Visualization 9: Top 10 Strongest Predictors of Customer Churn", use_container_width=True)
        else:
            st.info("Upload your 'feature_importance.png' to the repository root directory to visualize ML prediction weights.")
