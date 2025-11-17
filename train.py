import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_model():
    df = pd.read_csv("fake_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    feature_cols = [
        "uptime_24h_pct",
        "uptime_7d_pct",
        "total_downtime_24h_min",
        "total_downtime_7d_min",
        "downtime_events_24h",
        "avg_downtime_duration_min"
    ]

    X = df[feature_cols]
    y = df["failure_in_next_24h"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)]
    )

    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred_class))

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": model,
        "features": feature_cols
    }, "models/lgbm_failure_model.pkl")
    print("Model saved to models/lgbm_failure_model.pkl")

if __name__ == "__main__":
    train_model()