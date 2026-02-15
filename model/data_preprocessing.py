import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):

    df = pd.read_csv(path)

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y
