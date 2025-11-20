from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

def detect_flapping_events(histories: List[Dict], threshold_min: float = 1.0) -> List[Dict]:
    if len(histories) < 2:
        return []
    sorted_hist = sorted(histories, key=lambda x: x['createdAt'])
    events = []
    for i in range(1, len(sorted_hist)):
        prev, curr = sorted_hist[i-1], sorted_hist[i]
        dt_min = (curr['createdAt'] - prev['createdAt']).total_seconds() / 60.0
        if prev['status'] != curr['status'] and dt_min <= threshold_min:
            events.append({
                'device_id': curr['deviceId'],
                'timestamp': curr['createdAt'],
                'duration_min': dt_min
            })
    return events

def build_downtime_segments(histories: List[Dict]) -> List[Dict]:
    segments = []
    offline_start = None
    for h in histories:
        if h['status'] == 'OFFLINE' and offline_start is None:
            offline_start = h['createdAt']
        elif h['status'] == 'ONLINE' and offline_start is not None:
            duration = (h['createdAt'] - offline_start).total_seconds() / 60.0
            if duration > 0:
                segments.append({
                    'device_id': h['deviceId'],
                    'start': offline_start,
                    'end': h['createdAt'],
                    'duration_min': duration
                })
            offline_start = None
    if offline_start is not None:
        duration = (datetime.utcnow() - offline_start).total_seconds() / 60.0
        if duration > 0:
            segments.append({
                'device_id': histories[-1]['deviceId'],
                'start': offline_start,
                'end': datetime.utcnow(),
                'duration_min': duration
            })
    return segments

def calculate_flapping_metrics(histories: List[Dict], snapshot_time: datetime, lookback_hours: int) -> Dict[str, Any]:
    start = snapshot_time - timedelta(hours=lookback_hours)
    recent = [h for h in histories if start <= h['createdAt'] <= snapshot_time]
    
    is_24h = lookback_hours <= 48
        
    if len(recent) < 2:
        return {
            "flapping_events_24h" if is_24h else "flapping_events_7d": 0,
            "flapping_intensity_24h" if is_24h else "flapping_intensity_7d": 0.0,
            "avg_flap_duration_min" if is_24h else "avg_flap_duration_7d_min": 0.0,
            "stability_score_24h" if is_24h else "stability_score_7d": 100.0,
        }

    flapping = detect_flapping_events(recent, threshold_min=1.0)
    count = len(flapping)
    intensity = count / lookback_hours

    transitions = []
    for i in range(1, len(recent)):
        if recent[i]['status'] != recent[i-1]['status']:
            dt = (recent[i]['createdAt'] - recent[i-1]['createdAt']).total_seconds() / 60.0
            transitions.append(max(0.1, dt))

    avg_duration = np.mean(transitions) if transitions else 999.0
    score = 100.0

    if count > 0:
        score *= max(0.01, 1.0 - intensity / (10 if lookback_hours == 24 else 5.0))
    if avg_duration < (30 if lookback_hours == 24 else 60):
        score *= max(0.1, avg_duration / (30))
    if  intensity > 2.0:
        score *= 0.5

    if is_24h:
        return {
            "flapping_events_24h": count,
            "flapping_intensity_24h": round(intensity, 3),
            "avg_flap_duration_min": round(avg_duration, 2),       
            "stability_score_24h": round(max(0.0, min(100.0, score)), 2)
        }
    else:
        return {
            "flapping_events_7d": count,
            "flapping_intensity_7d": round(intensity, 3),
            "avg_flap_duration_7d_min": round(avg_duration, 2),   
            "stability_score_7d": round(max(0.0, min(100.0, score)), 2)
        }