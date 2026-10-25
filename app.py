"""
Bhutan Commodity Export Potential — Machine Learning Dashboard

GitHub setup:
1. Save this file as app.py.
2. Upload your data file as export_data.csv to GitHub and copy its RAW URL.
3. Update the GITHUB_RAW_URL variable below with your actual link.
4. Create requirements.txt containing:
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
5. Run deployment or launch locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Page settings
st.set_page_config(
    page_title="Bhutan Export Potential ML Dashboard",
    page_icon="🇧🇹",
    layout="wide"
)

TARGET = "high_potential"
POSITIVE = "High Potential"
NEGATIVE = "Standard Growth"

@st.cache_data
def get_clean_dataset():
 
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/anilchhetri-47/Export-in-Bhutan/refs/heads/main/export_dataset_clean%20(1).csv"
    
    try:
        # Pull live data file stream directly from your repository
        df = pd.read_csv(GITHUB_RAW_URL)
    except Exception as e:
        st.error(f"Failed to fetch data from GitHub. Please check your GITHUB_RAW_URL connection string. Error: {e}")
        st.stop()
    
    # Strip invisible spaces, BOM characters, or quotes from column headers
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    # Identify target columns using flexible text matching to prevent KeyErrors
    qty_col = [c for c in df.columns if 'qty' in c.lower()]
    val_col = [c for c in df.columns if 'val' in c.lower()]
    
    qty_name = qty_col[0] if qty_col else 'Qty'
    val_name = val_col[0] if val_col else 'Value'
    
    # Fill empty text columns and convert quantity and numeric inputs safely
    if qty_name in df.columns:
        df['Qty_Cleaned'] = pd.to_numeric(df[qty_name], errors='coerce')
        df['Qty_Cleaned'] = df['Qty_Cleaned'].fillna(df['Qty_Cleaned'].median() if df['Qty_Cleaned'].notna().any() else 5000)
    else:
        df['Qty_Cleaned'] = 5000

    if val_name in df.columns:
        df['Value_Cleaned'] = pd.to_numeric(df[val_name], errors='coerce')
        df['Value_Cleaned'] = df['Value_Cleaned'].fillna(df['Value_Cleaned'].median() if df['Value_Cleaned'].notna().any() else 500000)
    else:
        df['Value_Cleaned'] = 500000
    
    # Enrich with synthetic pipeline features for predicting high export potential
    rng = np.random.default_rng(42)
    n = len(df)
    
    df['global_demand_growth'] = rng.uniform(0.5, 9.5, n).round(1)
    df['tariff_advantage_score'] = rng.uniform(1, 10, n).round(1)
    df['logistical_cost_score'] = rng.uniform(1, 10, n).round(1)
    
    # Formula logic for assigning target classifications
    latent = (0.4 * np.log1p(df['Value_Cleaned']) + 0.35 * df['global_demand_growth'] + 
              0.25 * df['tariff_advantage_score'] - 0.3 * df['logistical_cost_score'] + 
              rng.normal(0, 1.0, n))
    
    threshold = np.quantile(latent, 0.65)
    df[TARGET] = np.where(latent >= threshold, POSITIVE, NEGATIVE)
    return df

@st.cache_resource
def run_model_training(data):
    X = data[['Qty_Cleaned', 'Value_Cleaned', 'global_demand_growth', 'tariff_advantage_score', 'logistical_cost_score']]
    y = data[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    pos_idx = list(model.classes_).index(POSITIVE)
    probs = model.predict_proba(X_test_scaled)[:, pos_idx]
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score((y_test == POSITIVE).astype(int), probs),
        "matrix": confusion_matrix(y_test, preds, labels=model.classes_),
        "report": classification_report(y_test, preds, output_dict=True),
        "classes": list(model.classes_)
    }
    return model, scaler, metrics

# Main application runtime execution
df = get_clean_dataset()
model, scaler, metrics = run_model_training(df)

st.title("Bhutan Commodity Export Potential Analysis")
st.caption("A live machine learning tracking report reading structured custom files directly from GitHub.")
st.warning(
    "This Report is not an official assessment of an individual, "
    "private entities or government agencies. It should not be "
    "used for disciplinary, employment, procurement, or legal decisions. "
    "The current dataset is synthetic and does not represent actual "
    "Bhutanese project records."
)

overview_tab, data_tab, prediction_tab, results_tab = st.tabs([
    "📈 Overview", "🔍 Explore Data", "🔮 Prediction Demo", "📊 Model Results"
])

with overview_tab:
    st.subheader("Project Objectives")
    st.write(
        "This system processes Bhutan Trade Statistics 2025 records from Bhutan to discover Bhutan's export patterns to India. "
        "Using **Logistic Regression**, it determines if a chosen commodity exhibits properties aligning with high growth sectors."
    )
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Commodity Catalog Entries", f"{len(df):,}")
    c2.metric("Pipeline Architecture", "Logistic Regression")
    c3.metric("Class Tuning Weights", "Balanced Scale")
    
    st.markdown("### Process Engineering Workflow")
    st.info("1. Pull Export Dataset ➔ 2. Splite Data ➔ 3. Preprocess Variables ➔ Train Model ➔ 5. Evaluate model ➔ 6. Generate Result")
    st.info(
        "A real Bhutanese study would require appropriately collected "
        "project records, clearly defined indicators, expert validation, "
        "ethical/data-governance review where applicable, and testing "
        "against real outcomes before the model could be used for "
        "Export prediction and decision support."
    )

with data_tab:
    st.subheader("Exploratory Data Analysis (EDA)")
    
    metric_view = st.selectbox(
        "Select Independent Metric Target:",
        ["Value_Cleaned", "Qty_Cleaned", "global_demand_growth", "tariff_advantage_score", "logistical_cost_score"]
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x=metric_view, hue=TARGET, multiple="stack", palette="copper", ax=ax, kde=True)
    ax.set_title(f"Distribution Vector of {metric_view.replace('_', ' ').title()}")
    st.pyplot(fig)
    
    st.markdown("### Active Repo Dataset Framework")
    st.dataframe(df, use_container_width=True)

with prediction_tab:
    st.subheader("Commodity Potential Simulator")
    st.write("Adjust configuration profiles below to calculate hypothetical trade indicators against the network.")
    
    col1, col2 = st.columns(2)
    with col1:
        qty_in = st.number_input("Export Volume (Units)", min_value=1.0, value=50000.0)
        val_in = st.number_input("Total Declared Value (BTN)", min_value=1.0, value=2500000.0)
        demand_in = st.slider("Global Market Demand Velocity", 0.0, 10.0, 5.5)
    with col2:
        tariff_in = st.slider("Tariff Exempt Advantage Rating", 1.0, 10.0, 6.0)
        logistic_in = st.slider("Logistical Friction/Freight Cost", 1.0, 10.0, 4.0)
        
    if st.button("Evaluate Export Status"):
        input_data = pd.DataFrame([[qty_in, val_in, demand_in, tariff_in, logistic_in]],
                                  columns=['Qty_Cleaned', 'Value_Cleaned', 'global_demand_growth', 'tariff_advantage_score', 'logistical_cost_score'])
        
        scaled_input = scaler.transform(input_data)
        
        # 🌟 CRITICAL FIX: Cast prediction and probability explicitly to Python scalars
        prediction_val = str(model.predict(scaled_input)[0])
        pos_idx = metrics["classes"].index(POSITIVE)
        probability_val = float(model.predict_proba(scaled_input)[0][pos_idx])
        
        st.markdown("---")
        if prediction_val == POSITIVE:
            st.success(f"🎉 **Status:** {prediction_val} (Confidence Vector: {probability_val:.1%})")
        else:
            st.error(f"⚠️ **Status:** {prediction_val} (Confidence Vector: {probability_val:.1%})")

with results_tab:
    st.subheader("Logistic Regression Validation Matrix")
    
    r1, r2 = st.columns(2)
    r1.metric("Calculated Test Accuracy Score", f"{metrics['accuracy']:.1%}")
    r2.metric("ROC-AUC Score Baseline", f"{metrics['roc_auc']:.2f}")
    
    st.markdown("### Classification Evaluation Matrix Breakdown")
    st.dataframe(pd.DataFrame(metrics["report"]).transpose(), use_container_width=True)
    
    st.markdown("### Confusion Matrix Output Visualizer")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
    sns.heatmap(metrics["matrix"], annot=True, fmt="d", xticklabels=metrics["classes"], yticklabels=metrics["classes"], cmap="Oranges", ax=ax_cm)
    ax_cm.set_xlabel("Predicted Categories")
    ax_cm.set_ylabel("True Ground Labels")
    st.pyplot(fig_cm)

st.divider()
st.caption(
    "Bhutan India Export AI Project | Synthetic Data | Educational/Research Prototype | By: Anil Chhetri | OCT 2026")
