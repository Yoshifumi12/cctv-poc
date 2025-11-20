from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from prisma import Prisma
from datetime import datetime, timedelta, timezone
from predict import predict_failure_probability
import numpy as np
from common.metrics import calculate_flapping_metrics

app = FastAPI()
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
    uptime_24h_pct: float
    total_downtime_24h_min: float
    downtime_events_24h: int
    avg_downtime_duration_24h_min: float
    flapping_events_24h: int
    flapping_intensity_24h: float
    avg_flap_duration_min: float
    stability_score_24h: float
    uptime_7d_pct: float
    total_downtime_7d_min: float
    downtime_events_7d: int
    avg_downtime_duration_7d_min: float
    flapping_events_7d: int
    flapping_intensity_7d: float
    avg_flap_duration_7d_min: float
    stability_score_7d: float

CCTV_BRANDS = ["HIKVISION", "DAHUA", "ACTI", "AXIS", "BOSCH", "PANASONIC", "SONY", "SAMSUNG", "GKB", "MARCH", "TRUVISION"]

def calculate_reliability_score(stats: Dict[str, Any]) -> float:
    score = 100.0
    score *= (stats.get("uptime_24h_pct", 100) / 100)
    score *= (stats.get("uptime_7d_pct", 100) / 100) ** 0.5
    score *= (stats.get("stability_score_24h", 100) / 100) ** 0.7
    score *= (stats.get("stability_score_7d", 100) / 100) ** 0.5
    flaps = stats.get("flapping_events_24h", 0)
    if flaps > 20:
        score *= 0.2
    elif flaps > 10:
        score *= 0.4
    return max(0.0, min(100.0, round(score, 2)))

def get_risk_level(prob: float) -> str:
    if prob >= 0.8: return "CRITICAL"
    if prob >= 0.6: return "HIGH"
    if prob >= 0.4: return "MEDIUM"
    if prob >= 0.2: return "LOW"
    return "VERY_LOW"

async def compute_device_stats(histories: List[Any]) -> Dict[str, Any]:
    if not histories:
        return None

    now = datetime.now(timezone.utc)
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)

    recent = sorted([h for h in histories if h.createdAt >= week_ago], key=lambda x: x.createdAt)

    downtime_24h = downtime_7d = events_24h = events_7d = 0.0
    durations_24h = durations_7d = []

    offline_start = None
    for h in recent:
        if h.status == "OFFLINE" and offline_start is None:
            offline_start = h.createdAt
        elif h.status == "ONLINE" and offline_start is not None:
            duration = (h.createdAt - offline_start).total_seconds() / 60.0
            downtime_7d += duration
            durations_7d.append(duration)
            events_7d += 1
            if h.createdAt >= day_ago:
                downtime_24h += duration
                durations_24h.append(duration)
                events_24h += 1
            offline_start = None

    if offline_start:
        duration = (now - offline_start).total_seconds() / 60.0
        if duration > 0:
            downtime_7d += duration
            durations_7d.append(duration)
            events_7d += 1
            if now >= day_ago:
                downtime_24h += duration
                durations_24h.append(duration)
                events_24h += 1

    uptime_24h = max(0.0, min(100.0, (1440 - downtime_24h) / 1440 * 100))
    uptime_7d = max(0.0, min(100.0, (10080 - downtime_7d) / 10080 * 100))

    flapping_24h = calculate_flapping_metrics([h.__dict__ for h in histories], now, 24)
    flapping_7d = calculate_flapping_metrics([h.__dict__ for h in histories], now, 7*24)

    return {
        "uptime_24h_pct": round(uptime_24h, 2),
        "uptime_7d_pct": round(uptime_7d, 2),
        "total_downtime_24h_min": round(downtime_24h, 2),
        "total_downtime_7d_min": round(downtime_7d, 2),
        "downtime_events_24h": int(events_24h),
        "downtime_events_7d": int(events_7d),
        "avg_downtime_duration_24h_min": round(np.mean(durations_24h) if durations_24h else 0.0, 2),
        "avg_downtime_duration_7d_min": round(np.mean(durations_7d) if durations_7d else 0.0, 2),
        "status": recent[-1].status if recent else "UNKNOWN",
        **flapping_24h,
        **flapping_7d
    }

async def get_all_device_stats() -> List[Dict]:
    await db.connect()
    try:
        devices = await db.device.find_many(
            where={"brand": {"in": CCTV_BRANDS}},
            include={"property": True, "statusHistories": {"order_by": {"createdAt": "desc"}, "take": 1000}}
        )
        result = []
        for dev in devices:
            if not dev.property or not dev.statusHistories:
                continue
            stats = await compute_device_stats(dev.statusHistories)
            if not stats:
                continue
            result.append({
                "device_id": dev.id,
                "ip_address": dev.ipAddress or "0.0.0.0",
                "name": dev.name or dev.ipAddress.split(".")[-1],
                "property_name": dev.property.name,
                **stats
            })
        return result
    finally:
        await db.disconnect()

@app.get("/api/prediction/devices/{device_id}", response_model=DevicePrediction)
async def get_device_prediction(device_id: str):
    await db.connect()
    try:
        device = await db.device.find_first(
            where={"id": device_id, "brand": {"in": CCTV_BRANDS}},
            include={"property": True, "statusHistories": {"take": 1000, "order_by": {"createdAt": "desc"}}}
        )
        if not device or not device.property:
            raise HTTPException(404, "Device not found")
        stats = await compute_device_stats(device.statusHistories)
        if not stats:
            raise HTTPException(404, "Insufficient data")
        prob = predict_failure_probability(stats)
        return DevicePrediction(
            device_id=device.id,
            ip_address=device.ipAddress,
            name=device.name,
            property_name=device.property.name,
            failure_probability=round(prob, 4),
            reliability_score=calculate_reliability_score(stats),
            risk_level=get_risk_level(prob),
            status=stats["status"],
            **{k: stats[k] for k in DevicePrediction.model_fields if k not in ["device_id","ip_address","name","property_name","failure_probability","reliability_score","risk_level","status"]}
        )
    finally:
        await db.disconnect()

@app.post("/api/prediction/devices/predictions/upsert")
async def upsert_predictions():
    stats_list = await get_all_device_stats()
    await db.connect()
    try:
        success = 0
        for dev in stats_list:
            try:
                prob = predict_failure_probability(dev)
                await db.deviceprediction.upsert(
                    where={"deviceId": dev["device_id"]},
                    data={
                        "create": {
                        "deviceId": dev["device_id"],
                        "failureProbability": round(prob, 6),
                        "reliabilityScore": calculate_reliability_score(dev),
                        "riskLevel": get_risk_level(prob),
                        **{k: dev.get(k.replace("24h", "24h").replace("7d", "7d")) or 0 for k in [
                            "uptime24h", "uptime7d", "totalDowntime24hMin", "totalDowntime7dMin",
                            "flappingEvents24h", "flappingEvents7d", "stabilityScore24h", "stabilityScore7d",
                            "downtimeEvents24h", "downtimeEvents7d", "avgDowntimeDuration24hMin", "avgDowntimeDuration7dMin",
                            "flappingIntensity24h", "flappingIntensity7d", "avgFlapDurationMin", "avgFlapDuration7dMin"
                        ]}
                    }, 
                        "update": {
                        "deviceId": dev["device_id"],
                        "failureProbability": round(prob, 6),
                        "reliabilityScore": calculate_reliability_score(dev),
                        "riskLevel": get_risk_level(prob),
                        **{k: dev.get(k.replace("24h", "24h").replace("7d", "7d")) or 0 for k in [
                            "uptime24h", "uptime7d", "totalDowntime24hMin", "totalDowntime7dMin",
                            "flappingEvents24h", "flappingEvents7d", "stabilityScore24h", "stabilityScore7d",
                            "downtimeEvents24h", "downtimeEvents7d", "avgDowntimeDuration24hMin", "avgDowntimeDuration7dMin",
                            "flappingIntensity24h", "flappingIntensity7d", "avgFlapDurationMin", "avgFlapDuration7dMin"
                        ]}
                    }}
                )
                success += 1
            except:
                pass
        return {"status": "success", "upserted": success, "total": len(stats_list)}
    finally:
        await db.disconnect()