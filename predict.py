import joblib
import pandas as pd

MODEL_PATH = "models/lgbm_failure_model.pkl"
_model_data = joblib.load(MODEL_PATH)
model = _model_data["model"]
features = _model_data["features"]

def predict_failure_probability(device_stats: dict) -> float:
    df = pd.DataFrame([device_stats])
    df = df[features]  
    prob = model.predict(df)[0]
    return float(prob)