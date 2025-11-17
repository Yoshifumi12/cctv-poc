from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from prisma import Prisma
from datetime import datetime, timedelta, timezone
from predict import predict_failure_probability

app = FastAPI(title="CCTV POC")

db = Prisma()

class DevicePrediction(BaseModel):
    device_id: str
    ip_address: str
    name: str
    property_name: str
    failure_probability: float
    reliability_score: float
    status: str
    risk_level: str
    uptime_24h: float
    uptime_7d: float

class DeviceStats(BaseModel):
    device_id: str
    ip_address: str
    name: str
    property_name: str
    uptime_24h_pct: float
    uptime_7d_pct: float
    total_downtime_24h_min: float
    total_downtime_7d_min: float
    downtime_events_24h: int
    avg_downtime_duration_min: float
    status: str
    failure_probability: Optional[float] = None
    reliability_score: Optional[float] = None
    risk_level: Optional[str] = None

CCTV_BRANDS = [
    "HIKVISION", "DAHUA", "ACTI", "AXIS", "BOSCH",
    "PANASONIC", "SONY", "SAMSUNG", "GKB", "MARCH", "TRUVISION"
]

def calculate_reliability_score(device_stats: Dict[str, Any]) -> float:
    score = 100.0 
    
    uptime_24h = device_stats.get("uptime_24h_pct", 100)
    score *= (uptime_24h / 100)
    
    downtime_events = device_stats.get("downtime_events_24h", 0)
    if downtime_events > 0:
        score *= max(0.5, 1 - (downtime_events * 0.1))

    avg_downtime = device_stats.get("avg_downtime_duration_min", 0)
    if avg_downtime > 30:  
        score *= max(0.3, 1 - (avg_downtime / 300))
    
    current_status = device_stats.get("status", "ONLINE")
    if current_status != "ONLINE":
        score *= 0.5
    
    return max(0, min(100, round(score, 2)))

def get_risk_level(failure_prob: float) -> str:
    if failure_prob >= 0.8:
        return "CRITICAL"
    elif failure_prob >= 0.6:
        return "HIGH"
    elif failure_prob >= 0.4:
        return "MEDIUM"
    elif failure_prob >= 0.2:
        return "LOW"
    else:
        return "VERY_LOW"

async def compute_device_uptime_stats(histories: List[Any]) -> Dict[str, Any]:
    if not histories:
        return None

    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)

    recent = [h for h in histories if h.createdAt >= week_ago]
    if not recent:
        return None

    recent = sorted(recent, key=lambda x: x.createdAt)

    downtime_min_24h = 0.0
    downtime_min_7d = 0.0
    events_24h = 0
    offline_start = None

    for i, h in enumerate(recent):
        ts = h.createdAt
        status = h.status

        if status == "OFFLINE":
            if offline_start is None:
                offline_start = ts
        else:
            if offline_start is not None:
                duration = (ts - offline_start).total_seconds() / 60.0
                downtime_min_7d += duration
                if ts >= day_ago:
                    downtime_min_24h += duration
                    events_24h += 1
                offline_start = None

        if i == len(recent) - 1 and offline_start is not None:
            duration = (now - offline_start).total_seconds() / 60.0
            downtime_min_7d += duration
            if now >= day_ago:
                downtime_min_24h += duration
                events_24h += 1

    total_min_24h = 24 * 60
    total_min_7d = 7 * 24 * 60

    uptime_24h_pct = max(0, min(100, (total_min_24h - downtime_min_24h) / total_min_24h * 100))
    uptime_7d_pct = max(0, min(100, (total_min_7d - downtime_min_7d) / total_min_7d * 100))
    avg_downtime = downtime_min_24h / max(events_24h, 1) if events_24h > 0 else 0

    latest_status = recent[-1].status

    return {
        "uptime_24h_pct": round(uptime_24h_pct, 2),
        "uptime_7d_pct": round(uptime_7d_pct, 2),
        "total_downtime_24h_min": round(downtime_min_24h, 2),
        "total_downtime_7d_min": round(downtime_min_7d, 2),
        "downtime_events_24h": events_24h,
        "avg_downtime_duration_min": round(avg_downtime, 2),
        "status": latest_status
    }

async def get_current_device_stats_from_db() -> List[Dict[str, Any]]:
    await db.connect()

    try:
        devices = await db.device.find_many(
            where={
                "brand": {"in": CCTV_BRANDS},
                "statusHistories": {"some": {}}
            },
            include={
                "property": True,
                "statusHistories": {
                    "order_by": {"createdAt": "desc"},
                    "take": 1000
                }
            }
        )

        stats_list = []

        for device in devices:
            if not device.property:
                continue

            uptime_stats = await compute_device_uptime_stats(device.statusHistories)
            if not uptime_stats:
                continue

            stats_list.append({
                "device_id": device.id,
                "ip_address": device.ipAddress,
                "name": device.name or f"CCTV-{device.ipAddress.split('.')[-1]}",
                "property_name": device.property.name,
                **uptime_stats
            })

        return stats_list

    except Exception as e:
        print(f"DB Error: {e}")
        raise e
    finally:
        await db.disconnect()

@app.get("/api/devices/at-risk", response_model=List[DevicePrediction])
async def get_at_risk_devices(threshold: float = Query(0.7, ge=0.0, le=1.0)):
    try:
        current_stats = await get_current_device_stats_from_db()
        at_risk = []

        for dev in current_stats:
            prob = predict_failure_probability(dev)
            reliability = calculate_reliability_score(dev)
            
            if prob > threshold:
                at_risk.append(DevicePrediction(
                    device_id=dev["device_id"],
                    ip_address=dev["ip_address"],
                    name=dev["name"],
                    property_name=dev["property_name"],
                    failure_probability=round(prob, 4),
                    reliability_score=reliability,
                    status=dev["status"],
                    risk_level=get_risk_level(prob),
                    uptime_24h=dev["uptime_24h_pct"],
                    uptime_7d=dev["uptime_7d_pct"]
                ))

        return sorted(at_risk, key=lambda x: x.failure_probability, reverse=True)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/devices/predictions", response_model=List[DeviceStats])
async def get_all_device_predictions():
    try:
        current_stats = await get_current_device_stats_from_db()
        predictions = []

        for dev in current_stats:
            prob = predict_failure_probability(dev)
            reliability = calculate_reliability_score(dev)
            
            predictions.append(DeviceStats(
                device_id=dev["device_id"],
                ip_address=dev["ip_address"],
                name=dev["name"],
                property_name=dev["property_name"],
                uptime_24h_pct=dev["uptime_24h_pct"],
                uptime_7d_pct=dev["uptime_7d_pct"],
                total_downtime_24h_min=dev["total_downtime_24h_min"],
                total_downtime_7d_min=dev["total_downtime_7d_min"],
                downtime_events_24h=dev["downtime_events_24h"],
                avg_downtime_duration_min=dev["avg_downtime_duration_min"],
                status=dev["status"],
                failure_probability=round(prob, 4),
                reliability_score=reliability,
                risk_level=get_risk_level(prob)
            ))

        return sorted(predictions, key=lambda x: x.failure_probability, reverse=True)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/devices/{device_id}/prediction")
async def get_device_prediction(device_id: str):
    try:
        await db.connect()
        
        device = await db.device.find_first(
            where={
                "id": device_id,
                "brand": {"in": CCTV_BRANDS}
            },
            include={
                "property": True,
                "statusHistories": {
                    "order_by": {"createdAt": "desc"},
                    "take": 1000
                }
            }
        )
        
        if not device or not device.property:
            raise HTTPException(status_code=404, detail="Device not found")
        
        uptime_stats = await compute_device_uptime_stats(device.statusHistories)
        if not uptime_stats:
            raise HTTPException(status_code=404, detail="Insufficient data for prediction")
        
        prob = predict_failure_probability(uptime_stats)
        reliability = calculate_reliability_score(uptime_stats)
        
        return {
            "device_id": device.id,
            "ip_address": device.ipAddress,
            "name": device.name,
            "property_name": device.property.name,
            "failure_probability": round(prob, 4),
            "reliability_score": reliability,
            "risk_level": get_risk_level(prob),
            **uptime_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        await db.disconnect()

@app.get("/health")
async def health():
    try:
        await db.connect()
        count = await db.device.count()
        await db.disconnect()
        return {"status": "healthy", "devices_in_db": count}
    except Exception as e:
        return {"status": "db_unreachable", "error": str(e)}