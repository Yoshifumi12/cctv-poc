from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import random
from predict import predict_failure_probability

app = FastAPI(title="CCTV POC")

class DevicePrediction(BaseModel):
    device_id: str
    ip_address: str
    name: str
    property_name: str
    failure_probability: float
    status: str

def get_current_device_stats():
    df = pd.read_csv("fake_data.csv")
    latest = df.sort_values("timestamp").groupby("device_id").tail(1)
    
    devices = []
    for _, row in latest.iterrows():
        devices.append({
            "device_id": row["device_id"],
            "ip_address": row["ip_address"],
            "name": f"CCTV-{row['ip_address'].split('.')[-1]}",
            "property_name": f"Property-{random.randint(1, 10)}",
            "uptime_24h_pct": row["uptime_24h_pct"],
            "uptime_7d_pct": row["uptime_7d_pct"],
            "total_downtime_24h_min": row["total_downtime_24h_min"],
            "total_downtime_7d_min": row["total_downtime_7d_min"],
            "downtime_events_24h": row["downtime_events_24h"],
            "avg_downtime_duration_min": row["avg_downtime_duration_min"],
            "status": row["status"]
        })
    return devices

@app.get("/api/devices/at-risk", response_model=List[DevicePrediction])
async def get_at_risk_devices(threshold: float = 0.7):
    try:
        current_stats = get_current_device_stats()
        at_risk = []

        for dev in current_stats:
            prob = predict_failure_probability(dev)
            if prob > threshold:
                at_risk.append(DevicePrediction(
                    device_id=dev["device_id"],
                    ip_address=dev["ip_address"],
                    name=dev["name"],
                    property_name=dev["property_name"],
                    failure_probability=round(prob, 4),
                    status=dev["status"]
                ))

        return sorted(at_risk, key=lambda x: x.failure_probability, reverse=True)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "CCTV Failure Prediction POC - /api/devices/at-risk"}