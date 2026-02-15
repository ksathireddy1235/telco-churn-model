import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

st.set_page_config(page_title="📡 Churn Intelligence Hub", layout="wide")

# =========================================
# HEADER
# =========================================
st.title("📡 Churn Intelligence Hub")
st.markdown("### 🚀 Telco Customer Churn – End-to-End ML System")
st.markdown("---")

# =========================================
# LOAD MODELS & FILES
# =========================================
MODEL_DIR = "model"

models = {
    "🧠 Logistic Regression": joblib.load(f"{MODEL_DIR}/Logistic_Regression.pkl"),
    "🌳 Decision Tree": joblib.load(f"{MODEL_DIR}/Decision_Tree.pkl"),
    "📍 KNN": joblib.load(f"{MODEL_DIR}/KNN.pkl"),
    "📊 Naive Bayes": joblib.load(f"{MODEL_DIR}/Naive_Bayes.pkl"),
    "🌲 Random Forest": joblib.load(f"{MODEL_DIR}/Random_Forest.pkl"),
    "⚡ XGBoost": joblib.load(f"{MODEL_DIR}/XGBoost.pkl"),
}

scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
features = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
metrics_df = pd.read_csv(f"{MODEL_DIR}/model_metrics.csv")

# =========================================
# TABS
# =========================================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "🏆 Model Scoreboard", "📊 Comparison Charts"])

# =========================================
# TAB 1 – PREDICTION
# =========================================
with tab1:

    uploaded = st.file_uploader("📂 Upload Telco CSV File", type="csv")
    model_choice = st.selectbox("🧩 Choose Machine Learning Model", list(models.keys()))

    if uploaded:
        df = pd.read_csv(uploaded)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Preprocessing
        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

        y_true = None
        if "Churn" in df.columns:
            y_true = df["Churn"].map({"Yes": 1, "No": 0})
            df.drop("Churn", axis=1, inplace=True)

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category").cat.codes

        for col in features:
            if col not in df.columns:
                df[col] = 0

        df = df[features]
        df_scaled = scaler.transform(df)

        model = models[model_choice]

        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        # ===============================
        # REQUIRED 6 METRICS DISPLAY
        # ===============================
        if y_true is not None:

            accuracy = accuracy_score(y_true, preds)
            precision = precision_score(y_true, preds)
            recall = recall_score(y_true, preds)
            f1 = f1_score(y_true, preds)
            mcc = matthews_corrcoef(y_true, preds)
            auc_score = roc_auc_score(y_true, probs)

            st.markdown("## 📊 Model Evaluation Metrics")

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("🎯 Accuracy", f"{accuracy:.4f}")
            col2.metric("📈 AUC Score", f"{auc_score:.4f}")
            col3.metric("🔍 Precision", f"{precision:.4f}")

            col4.metric("📢 Recall", f"{recall:.4f}")
            col5.metric("📊 F1 Score", f"{f1:.4f}")
            col6.metric("🧮 MCC", f"{mcc:.4f}")

            st.markdown("---")

        # KPI Cards
        colA, colB, colC = st.columns(3)
        colA.metric("📊 Total Records", len(df))
        colB.metric("⚠️ Predicted Churn", (preds == 1).sum())
        colC.metric("📈 Avg Probability", f"{probs.mean():.2%}")

        # Predictions Table
        results = pd.DataFrame({
            "Prediction": ["Churn 🔴" if x == 1 else "No Churn 🟢" for x in preds],
            "Probability": probs
        })

        st.subheader("📊 Prediction Results")
        st.dataframe(results, use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions",
            results.to_csv(index=False),
            "predictions.csv"
        )

        # Probability Distribution
        st.subheader("📈 Probability Distribution")
        fig, ax = plt.subplots()
        sns.histplot(probs, bins=20, kde=True, ax=ax)
        st.pyplot(fig)

        # Confusion Matrix
        if y_true is not None:
            st.subheader("📌 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_true, preds),
                        annot=True, fmt="d",
                        cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)

            st.subheader("📈 ROC Curve")
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.legend()
            st.pyplot(fig)

# =========================================
# TAB 2 – MODEL SCOREBOARD
# =========================================
with tab2:

    st.subheader("🏆 Model Performance Summary")

    numeric_cols = metrics_df.select_dtypes(include="number").columns

    styled_metrics = (
        metrics_df.style
        .background_gradient(subset=numeric_cols, cmap="PuBuGn")
        .format({col: "{:.4f}" for col in numeric_cols})
    )

    st.dataframe(styled_metrics, use_container_width=True)

    best_model = metrics_df.loc[metrics_df["MCC"].idxmax()]
    st.success(f"🥇 Best Model (Based on MCC): {best_model['Model']}")

# =========================================
# TAB 3 – COMPARISON CHARTS
# =========================================
with tab3:

    st.subheader("📊 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="Model", y="Accuracy", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("📊 F1 Score Comparison")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="Model", y="F1_Score", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("Developed for ML Assignment 2 | 📡 Churn Intelligence Hub")
