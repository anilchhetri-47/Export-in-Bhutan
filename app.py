import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Bhutan Export Analytics",
    page_icon="🇧🇹",
    layout="wide"
)

st.title("🇧🇹 Bhutan Export Analytics to India")
st.markdown(
    """
    AI-based analysis of Bhutanese exports to India using Logistic Regression.
    """
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Export Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    overview_tab, data_tab, prediction_tab, results_tab = st.tabs(
    ["Overview", "Explore data", "Prediction demo", "Model results"]
)
    
 overview_tab, data_tab, prediction_tab, results_tab = st.tabs(
    ["Overview", "Explore data", "Prediction demo", "Model results"]
     
with overview_tab:
    st.subheader("Project objective")
    st.write(
        "This app shows how logistic regression can identify patterns "
        "associated with commodities exported to India from Bhutan. The data "
        "were Taken from Bhutan Trade Statistics 2025"
        "from MoICE website."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Fictional records", f"{len(data):,}")
    col2.metric("Input features", len(data.columns) - 1)
    col3.metric("Algorithm", "Logistic regression")

    st.markdown("### Machine-learning workflow")
    st.write(
        "1. uploaded data → 2. Split data → 3. Preprocess variables "
        "→ 4. Train model → 5. Evaluate model → 6. Display results"
    )
    st.info(
        "This dataset has been modified based on my project needs and "
        "this data analysis is just for project purpose only. "
        
    )

with data_tab:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # CLEAN DATA
    # ---------------------------------------------------

    df = df.dropna()

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["Value"] = pd.to_numeric(
        df["Value"],
        errors="coerce"
    )

    df = df.dropna()

    # ---------------------------------------------------
    # TARGET VARIABLE
    # ---------------------------------------------------

    median_value = df["Value"].median()

    df["High_Potential"] = (
        df["Value"] > median_value
    ).astype(int)

    # ---------------------------------------------------
    # FEATURES
    # ---------------------------------------------------

    X = df[["Quantity", "Value"]]

    y = df["High_Potential"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    
    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
with prediction_tab:
    model = LogisticRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # ---------------------------------------------------
    # DASHBOARD METRICS
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Commodities",
        len(df)
    )

    col2.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

    col3.metric(
        "Total Export Value",
        f"Nu {df['Value'].sum():,.0f}"
    )

    st.markdown("---")

    # ---------------------------------------------------
    # TOP 10 EXPORTED COMMODITIES
    # ---------------------------------------------------

    st.header("📈 Top Exported Commodities")

    top_exports = (
        df.sort_values(
            "Value",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        top_exports,
        x="Value",
        y="Commodity",
        orientation="h",
        color="Value",
        title="Top Commodities by Export Value"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------------------
    # HIGH POTENTIAL SCORE
    # ---------------------------------------------------

    df["Potential_Score"] = (
        model.predict_proba(X_scaled)[:, 1]
    )

    ranking = df.sort_values(
        "Potential_Score",
        ascending=False
    )

    st.header("🏆 Highest Export Potential Commodity")

    best = ranking.iloc[0]

    st.success(
        f"""
        Commodity: {best['Commodity']}

        Quantity: {best['Quantity']:,.0f}

        Export Value: Nu {best['Value']:,.0f}

        Potential Score: {best['Potential_Score']:.2%}
        """
    )

    # ---------------------------------------------------
    # TOP POTENTIAL TABLE
    # ---------------------------------------------------

    st.header("📊 Commodity Potential Ranking")

    st.dataframe(
        ranking[
            [
                "Commodity",
                "Quantity",
                "Value",
                "Potential_Score"
            ]
        ]
        .head(15)
    )

    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------
with results_tab:
    st.header("📌 Feature Importance")

    importance = pd.DataFrame({
        "Feature": [
            "Quantity",
            "Value"
        ],
        "Coefficient": model.coef_[0]
    })

    st.dataframe(
        importance
    )

    fig2 = px.bar(
        importance,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Coefficient"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------

    st.header("🎯 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig3, ax = plt.subplots(
        figsize=(5,4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig3)

    # ---------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------

    st.header("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # ---------------------------------------------------
    # PREDICTION TOOL
    # ---------------------------------------------------

    st.header("🤖 Predict Commodity Potential")

    qty = st.number_input(
        "Quantity",
        min_value=0.0,
        value=1000.0
    )

    value = st.number_input(
        "Value",
        min_value=0.0,
        value=50000.0
    )

    if st.button("Predict"):

        sample = scaler.transform(
            [[qty, value]]
        )

        prediction = model.predict(sample)

        probability = (
            model.predict_proba(sample)[0][1]
        )

        if prediction[0] == 1:

            st.success(
                f"""
                HIGH EXPORT POTENTIAL ✅

                Probability:
                {probability:.2%}
                """
            )

        else:

            st.error(
                f"""
                LOW EXPORT POTENTIAL ❌

                Probability:
                {probability:.2%}
                """
            )

    # ---------------------------------------------------
    # EXPLANATION
    # ---------------------------------------------------

    st.header("📚 Why Logistic Regression?")

    st.info(
        """
        Logistic Regression is a supervised machine learning algorithm
        that is commonly used for binary classification.

        In this project, commodities are classified as:

        • High Export Potential

        • Low Export Potential

        based on their export quantity and export value.

        The model calculates the probability that a commodity belongs
        to the High Export Potential class.

        Advantages:

        ✅ Easy to understand

        ✅ Fast to train

        ✅ Suitable for classification tasks

        ✅ Generates probability scores

        ✅ Useful for ranking export commodities
        """
    )

   # else:

    st.info(
        """
        Upload your cleaned CSV file with columns:

        Commodity, Quantity, Value
        """
    )
