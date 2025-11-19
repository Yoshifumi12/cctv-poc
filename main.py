from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from prisma import Prisma
from datetime import datetime, timedelta, timezone
from predict import predict_failure_probability
import numpy as np

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
    total_downtime_24h_min: float
    downtime_events_24h: int
    avg_downtime_duration_24h_min: float
    flapping_events_24h: int
    flapping_intensity_24h: float
    avg_flap_duration_min: float
    stability_score_24h: float
    uptime_7d: float 
    total_downtime_7d_min: float 
    downtime_events_7d: int 
    avg_downtime_duration_7d_min: float 
    flapping_events_7d: int 
    flapping_intensity_7d: float 
    avg_flap_duration_7d_min: float 
    stability_score_7d: float 
    
CCTV_BRANDS = [
    "HIKVISION", "DAHUA", "ACTI", "AXIS", "BOSCH",
    "PANASONIC", "SONY", "SAMSUNG", "GKB", "MARCH", "TRUVISION"
]

def detect_flapping_events(histories, flapping_threshold_min=1.0):
    if len(histories) < 2:
        return []

    flapping_events = []
    histories_sorted = sorted(histories, key=lambda x: x.createdAt)

    for i in range(1, len(histories_sorted)):
        prev = histories_sorted[i-1]
        curr = histories_sorted[i]
        time_diff_min = (curr.createdAt - prev.createdAt).total_seconds() / 60.0

        if prev.status != curr.status and time_diff_min <= flapping_threshold_min:
            flapping_events.append({
                'device_id': curr.deviceId,
                'timestamp': curr.createdAt,
                'time_since_last_change_min': time_diff_min,
                'is_flap': True
            })

    return flapping_events

def calculate_flapping_metrics(device_histories, snapshot_time, lookback_hours=24):
    lookback_start = snapshot_time - timedelta(hours=lookback_hours)
    recent = [h for h in device_histories if lookback_start <= h.createdAt <= snapshot_time]
    
    if len(recent) < 2:
        return {
            "flapping_events_24h": 0,
            "flapping_intensity_24h": 0.0,
            "avg_flap_duration_min": 0.0,
            "stability_score_24h": 100.0
        }
        
    recent = sorted(recent, key=lambda x: x.createdAt)

    flapping_events = []
    change_intervals = []

    for i in range(1, len(recent)):
        prev, curr = recent[i-1], recent[i]
        dt_min = (curr.createdAt - prev.createdAt).total_seconds() / 60.0

        if dt_min < 0:
            continue 

        if prev.status != curr.status:
            change_intervals.append(max(0.1, dt_min))  
            if dt_min <= 1.0: 
                flapping_events.append(curr)

    flapping_count = len(flapping_events)
    flapping_intensity = flapping_count / lookback_hours

    avg_flap_duration = np.mean(change_intervals) if change_intervals else 60.0  

    score = 100.0

    if flapping_intensity > 0:
        score *= max(0.1, 1.0 - (flapping_intensity / 10.0))

    if avg_flap_duration < 30:
        score *= max(0.2, avg_flap_duration / 30.0)

    score = max(0.0, min(100.0, score))

    return {
        "flapping_events_24h": flapping_count,
        "flapping_intensity_24h": round(flapping_intensity, 3),
        "avg_flap_duration_min": round(avg_flap_duration, 2),
        "stability_score_24h": round(score, 2)
    }


def calculate_flapping_metrics_7d(device_histories, snapshot_time):
    lookback_start = snapshot_time - timedelta(days=7)
    recent = [h for h in device_histories if lookback_start <= h.createdAt <= snapshot_time]
    
    if len(recent) < 2:
        return {
            "flapping_events_7d": 0,
            "flapping_intensity_7d": 0.0,
            "avg_flap_duration_7d_min": 0.0,
            "stability_score_7d": 100.0
        }

    recent = sorted(recent, key=lambda x: x.createdAt)

    flapping_events = []
    change_intervals = []

    for i in range(1, len(recent)):
        prev, curr = recent[i-1], recent[i]
        dt_min = (curr.createdAt - prev.createdAt).total_seconds() / 60.0

        if dt_min < 0:
            continue 

        if prev.status != curr.status:
            change_intervals.append(max(0.1, dt_min))
            if dt_min <= 1.0:
                flapping_events.append(curr)

    flapping_count = len(flapping_events)
    hours_in_7d = 7 * 24
    flapping_intensity = flapping_count / hours_in_7d

    avg_flap_duration = np.mean(change_intervals) if change_intervals else 120.0  

    score = 100.0
    if flapping_intensity > 0:
        score *= max(0.1, 1.0 - (flapping_intensity / 5.0))
    if avg_flap_duration < 60:
        score *= max(0.2, avg_flap_duration / 60.0)
    if flapping_intensity > 3.0:
        score *= 0.7

    score = max(0.0, min(100.0, score))

    return {
        "flapping_events_7d": flapping_count,
        "flapping_intensity_7d": round(flapping_intensity, 3),
        "avg_flap_duration_7d_min": round(avg_flap_duration, 2),
        "stability_score_7d": round(score, 2)
    }
        
def calculate_reliability_score(device_stats: Dict[str, Any]) -> float:
    score = 100.0

    score *= (device_stats.get("uptime_24h_pct", 100) / 100)
    score *= (device_stats.get("uptime_7d_pct", 100) / 100) ** 0.5  

    stab_24 = device_stats.get("stability_score_24h", 100)
    stab_7d = device_stats.get("stability_score_7d", 100)
    score *= (stab_24 / 100) ** 0.7
    score *= (stab_7d / 100) ** 0.5

    if device_stats.get("flapping_events_24h", 0) > 20:
        score *= 0.2
    elif device_stats.get("flapping_events_24h", 0) > 10:
        score *= 0.4

    return max(0.0, min(100.0, round(score, 2)))

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

    recent_histories = [h for h in histories if h.createdAt >= week_ago]
    if not recent_histories:
        return None

    recent_histories = sorted(recent_histories, key=lambda x: x.createdAt)

    downtime_min_24h = 0.0
    downtime_min_7d = 0.0
    downtime_events_24h = 0
    downtime_events_7d = 0
    downtime_durations_24h = []
    downtime_durations_7d = []

    offline_start = None

    for h in recent_histories:
        ts = h.createdAt
        status = h.status

        if status == "OFFLINE" and (offline_start is None):
            offline_start = ts

        elif status == "ONLINE" and offline_start is not None:
            duration_min = (ts - offline_start).total_seconds() / 60.0
            if duration_min < 0:  
                duration_min = 0.0

            downtime_min_7d += duration_min
            downtime_durations_7d.append(duration_min)
            downtime_events_7d += 1

            if ts >= day_ago:
                downtime_min_24h += duration_min
                downtime_durations_24h.append(duration_min)
                downtime_events_24h += 1

            offline_start = None  

    if offline_start is not None:
        duration_min = (now - offline_start).total_seconds() / 60.0
        if duration_min > 0:
            downtime_min_7d += duration_min
            downtime_durations_7d.append(duration_min)
            downtime_events_7d += 1

            if now >= day_ago:  
                downtime_min_24h += duration_min
                downtime_durations_24h.append(duration_min)
                downtime_events_24h += 1

    total_min_24h = 24 * 60 
    total_min_7d = 7 * 24 * 60  

    uptime_24h_pct = max(0.0, min(100.0, (total_min_24h - downtime_min_24h) / total_min_24h * 100))
    uptime_7d_pct = max(0.0, min(100.0, (total_min_7d - downtime_min_7d) / total_min_7d * 100))

    avg_downtime_24h = np.mean(downtime_durations_24h) if downtime_durations_24h else 0.0
    avg_downtime_7d = np.mean(downtime_durations_7d) if downtime_durations_7d else 0.0

    latest_status = recent_histories[-1].status if recent_histories else "UNKNOWN"

    flapping_24h = calculate_flapping_metrics(histories, now, lookback_hours=24)
    flapping_7d = calculate_flapping_metrics_7d(histories, now)

    return {
        "uptime_24h_pct": round(uptime_24h_pct, 2),
        "uptime_7d_pct": round(uptime_7d_pct, 2),
        "total_downtime_24h_min": round(downtime_min_24h, 2),
        "total_downtime_7d_min": round(downtime_min_7d, 2),
        "downtime_events_24h": downtime_events_24h,
        "downtime_events_7d": downtime_events_7d,
        "avg_downtime_duration_24h_min": round(avg_downtime_24h, 2),
        "avg_downtime_duration_7d_min": round(avg_downtime_7d, 2),
        "status": latest_status,
        **flapping_24h,
        **flapping_7d,
    }
async def get_current_device_stats_from_db() -> List[Dict[str, Any]]:
    await db.connect()

    try:
        print("DEBUG: Fetching devices from database...")
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

        print(f"DEBUG: Found {len(devices)} devices")
        stats_list = []

        for device in devices:
            print(f"DEBUG: Processing device: {device.id}, name: {device.name}, ip: {device.ipAddress}")
            
            if not device.property:
                print(f"DEBUG: Skipping device {device.id} - no property")
                continue

            if not device.statusHistories:
                print(f"DEBUG: Skipping device {device.id} - no status histories")
                continue

            print(f"DEBUG: Computing uptime stats for device {device.id}")
            uptime_stats = await compute_device_uptime_stats(device.statusHistories)
            if not uptime_stats:
                print(f"DEBUG: Skipping device {device.id} - no uptime stats")
                continue

            device_name = device.name or f"CCTV-{device.ipAddress.split('.')[-1]}" if device.ipAddress else "Unknown-Device"
            property_name = device.property.name or "Unknown-Property"
            ip_address = device.ipAddress or "0.0.0.0"
            
            print(f"DEBUG: Device {device.id} - name: '{device_name}', property: '{property_name}', ip: '{ip_address}'")

            stats_list.append({
                "device_id": device.id,
                "ip_address": ip_address,
                "name": device_name,
                "property_name": property_name,
                **uptime_stats
            })

        print(f"DEBUG: Returning {len(stats_list)} valid devices")
        return stats_list

    except Exception as e:
        print(f"DB Error: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        await db.disconnect()
        
@app.get("/api/prediction/devices/{device_id}", response_model=DevicePrediction)
async def get_device(device_id: str):
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
        risk_lvl = get_risk_level(prob)
        
        return DevicePrediction(
            device_id=device.id,
            ip_address=device.ipAddress,
            name=device.name,
            property_name=device.property.name,
            failure_probability=round(prob, 4),
            reliability_score=reliability,
            status=uptime_stats["status"],
            risk_level=risk_lvl,
            uptime_24h=uptime_stats["uptime_24h_pct"],
            uptime_7d=uptime_stats["uptime_7d_pct"],
            total_downtime_24h_min=uptime_stats["total_downtime_24h_min"],
            total_downtime_7d_min=uptime_stats["total_downtime_7d_min"],
            downtime_events_24h=uptime_stats["downtime_events_24h"],
            downtime_events_7d=uptime_stats["downtime_events_7d"],
            avg_downtime_duration_24h_min=uptime_stats["avg_downtime_duration_24h_min"],
            avg_downtime_duration_7d_min=uptime_stats["avg_downtime_duration_7d_min"],
            flapping_events_24h=uptime_stats["flapping_events_24h"],
            flapping_intensity_24h=uptime_stats["flapping_intensity_24h"],
            avg_flap_duration_min=uptime_stats["avg_flap_duration_min"],
            stability_score_24h=uptime_stats["stability_score_24h"],
            flapping_events_7d=uptime_stats["flapping_events_7d"],
            flapping_intensity_7d=uptime_stats["flapping_intensity_7d"],
            avg_flap_duration_7d_min=uptime_stats["avg_flap_duration_7d_min"],
            stability_score_7d=uptime_stats["stability_score_7d"],
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        await db.disconnect()
        
@app.post("/api/prediction/devices/predictions/upsert")
async def upsert_predictions():
    try:
        current_stats = await get_current_device_stats_from_db()
        total_devices = len(current_stats)
        success_count = 0
        failed_devices = []

        await db.connect()

        for idx, dev in enumerate(current_stats, start=1):
            device_id = dev["device_id"]

            try:
                # Safely calculate predictions
                prob = predict_failure_probability(dev)
                reliability = calculate_reliability_score(dev)
                risk_lvl = get_risk_level(prob)

                # Build prediction data with safe fallbacks
                pred_data = {
                    "deviceId": device_id,
                    "failureProbability": round(float(prob), 6),
                    "reliabilityScore": float(reliability),
                    "riskLevel": str(risk_lvl),
                    "uptime24h": float(dev.get("uptime_24h_pct") or 0.0),
                    "uptime7d": float(dev.get("uptime_7d_pct") or 0.0),
                    "flappingEvents24h": int(dev.get("flapping_events_24h") or 0),
                    "flappingEvents7d": int(dev.get("flapping_events_7d") or 0),
                    "stabilityScore24h": float(dev.get("stability_score_24h") or 0.0),
                    "stabilityScore7d": float(dev.get("stability_score_7d") or 0.0),
                    "totalDowntime24hMin": float(dev.get("total_downtime_24h_min") or 0.0),
                    "totalDowntime7dMin": float(dev.get("total_downtime_7d_min") or 0.0),
                    "downtimeEvents24h": int(dev.get("downtime_events_24h") or 0),
                    "downtimeEvents7d": int(dev.get("downtime_events_7d") or 0),
                    "avgDowntimeDuration24hMin": float(dev.get("avg_downtime_duration_24h_min") or 0.0),
                    "avgDowntimeDuration7dMin": float(dev.get("avg_downtime_duration_7d_min") or 0.0),
                    "flappingIntensity24h": float(dev.get("flapping_intensity_24h") or 0.0),
                    "flappingIntensity7d": float(dev.get("flapping_intensity_7d") or 0.0),
                    "avgFlapDurationMin": float(dev.get("avg_flap_duration_min") or 0.0),
                    "avgFlapDuration7dMin": float(dev.get("avg_flap_duration_7d_min") or 0.0),  # Fixed typo!
                }

                await db.deviceprediction.upsert(
                    where={"deviceId": device_id},
                    data={
                        "create": pred_data,
                        "update": pred_data,  # Same data for create & update (common pattern)
                    },
                )
                success_count += 1

                # Optional: log progress every 500 devices
                if idx % 500 == 0:
                    print(f"[UPSERT] Progress: {idx}/{total_devices} devices processed")

            except Exception as e:
                error_msg = str(e)
                failed_devices.append({"device_id": device_id, "error": error_msg})
                print(f"[UPSERT FAILED] Device {device_id} (#{idx}): {error_msg}")

        # Final result
        status = "success" if not failed_devices else "partial_success"
        result = {
            "status": status,
            "total_devices": total_devices,
            "upserted_count": success_count,
            "failed_count": len(failed_devices),
            "success_rate": f"{(success_count / total_devices * 100):.2f}%",
        }

        if failed_devices:
            result["failed_devices_sample"] = failed_devices[:20]  # First 20 errors
            result["note"] = f"{len(failed_devices)} devices failed. Check logs for details."

        print(f"[UPSERT COMPLETE] {success_count}/{total_devices} succeeded")
        return result

    except Exception as e:
        # This catches only unexpected crashes (e.g. DB connection lost)
        print(f"[CRITICAL ERROR] Upsert job failed completely: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction upsert failed: {str(e)}")
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