import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from data_preprocessing import load_and_preprocess
from logistic_model import build_model as logistic
from decision_tree_model import build_model as dt
from knn_model import build_model as knn
from naive_bayes_model import build_model as nb
from random_forest_model import build_model as rf
from xgboost_model import build_model as xgb


DATA_PATH = "../data/telco_churn.csv"

X, y = load_and_preprocess(DATA_PATH)

os.makedirs(".", exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

models = {
    "Logistic Regression": logistic(),
    "Decision Tree": dt(),
    "KNN": knn(),
    "Naive Bayes": nb(),
    "Random Forest": rf(),
    "XGBoost": xgb()
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1_Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")

pd.DataFrame(results).to_csv("model_metrics.csv", index=False)

print("🚀 All models trained successfully!")
