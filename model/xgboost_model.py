from xgboost import XGBClassifier

def build_model():
    return XGBClassifier(eval_metric="logloss")
