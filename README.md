# 📡 Telco Customer Churn Classification and Deployment

## 🧪 1. Project Objective

The goal of this project is to build and compare supervised machine learning models to predict customer churn for a telecommunications operator. Each model is evaluated using standardized performance metrics. A Streamlit web application is developed to interactively test models on new data.

---

## 📊 2. Problem Definition

Customer churn describes when a customer discontinues service. Predicting churn helps companies proactively retain at-risk customers. This project implements 6 ML models to classify churn based on customer demographics, service, and billing information.

---

## 📦 3. Dataset Overview

- **Dataset:** Telco Customer Churn  
- **Source:** Kaggle (manually downloaded)  
- **Total Samples:** 7,043+ rows  
- **Features:** 20+ columns including demographics and service usage  
- **Target Column:** `Churn` (Yes/No → binary class)

---

## 🧼 4. Data Preprocessing

The following steps are applied:

1. Drop irrelevant identifier column `customerID`.  
2. Convert `TotalCharges` to numeric datatype.  
3. Encode categorical features using Label Encoding.  
4. Map target (`Yes` → 1, `No` → 0).  
5. Handle missing values using median.  
6. Standardize features via `StandardScaler`.

---

## 🧠 5. Models Implemented

Six classification models are trained and evaluated:

| S.no | Model Name |
|------|------------|
| 1 | Logistic Regression |
| 2 | Decision Tree |
| 3 | K-Nearest Neighbors (KNN) |
| 4 | Naive Bayes |
| 5 | Random Forest |
| 6 | XGBoost |

Each model is defined in its own Python file under `model/`.

---

## 📏 6. Evaluation Metrics

All models are evaluated on:

✔ **Accuracy**  
✔ **AUC (Area Under ROC)**  
✔ **Precision**  
✔ **Recall**  
✔ **F1 Score**  
✔ **Matthews Correlation Coefficient (MCC)**

The evaluation results are stored in:

