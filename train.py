import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import asyncio
from datetime import datetime, timedelta
import asyncpg
import os
import numpy as np

FEATURE_COLS = [
    "uptime_24h_pct",
    "uptime_7d_pct",
    "total_downtime_24h_min",
    "total_downtime_7d_min",
    "downtime_events_24h",
    "downtime_events_7d",
    "avg_downtime_duration_24h_min",
    "avg_downtime_duration_7d_min",
    "flapping_events_24h",
    "flapping_intensity_24h",       
    "avg_flap_duration_min",
    "stability_score_24h",
    "stability_score_7d",
    "flapping_events_7d",           
    "flapping_intensity_7d",        
    "avg_flap_duration_7d_min",     
]

async def fetch_all_status_histories():
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='postgres',
        database='arms'
    )
    
    try:
        query = """
        SELECT 
            dsh.id,
            dsh."deviceId",
            dsh.status,
            dsh."createdAt",
            dsh.remarks,
            dsh."userId",
            d.name as device_name,
            d."ipAddress",
            p.name as property_name,
            p.location
        FROM "DeviceStatusHistory" dsh
        JOIN "Device" d ON dsh."deviceId" = d.id
        JOIN "Property" p ON d."propertyId" = p.id
        ORDER BY dsh."createdAt" ASC
        LIMIT 500000
        """
        
        rows = await conn.fetch(query)
        return rows
    finally:
        await conn.close()

def build_downtime_segments(histories):
    segments = []
    current_offline = None

    for h in histories:
        status = h['status']
        created_at = h['createdAt']
        device_id = h['deviceId']

        if status == "OFFLINE" and current_offline is None:
            current_offline = created_at
        elif status == "ONLINE" and current_offline is not None:
            duration = (created_at - current_offline).total_seconds() / 60.0
            if duration > 0:
                segments.append({
                    "device_id": device_id,
                    "start": current_offline,
                    "end": created_at,
                    "duration_min": duration
                })
            current_offline = None

    if current_offline is not None:
        now = datetime.utcnow()
        duration = (now - current_offline).total_seconds() / 60.0
        if duration > 0:
            segments.append({
                "device_id": h['deviceId'],
                "start": current_offline,
                "end": now,
                "duration_min": duration
            })

    flapping_events = detect_flapping_events(histories)
    
    return pd.DataFrame(segments), flapping_events

def detect_flapping_events(histories, flapping_threshold_min=10):
    if len(histories) < 2:
        return []

    flapping_events = []
    histories_sorted = sorted(histories, key=lambda x: x['createdAt'])

    for i in range(1, len(histories_sorted)):
        prev = histories_sorted[i-1]
        curr = histories_sorted[i]
        time_diff_min = (curr['createdAt'] - prev['createdAt']).total_seconds() / 60.0

        if prev['status'] != curr['status'] and time_diff_min <= flapping_threshold_min:
            flapping_events.append({
                'device_id': curr['deviceId'],
                'timestamp': curr['createdAt'],
                'time_since_last_change_min': time_diff_min,
                'is_flap': True
            })

    return flapping_events

def calculate_flapping_metrics_7d(device_histories, snapshot_time):
    lookback_start = snapshot_time - timedelta(days=7)
    recent_histories = [h for h in device_histories if lookback_start <= h['createdAt'] <= snapshot_time]

    if len(recent_histories) < 2:
        return {
            "flapping_events_7d": 0,
            "flapping_intensity_7d": 0.0,
            "avg_flap_duration_7d_min": 0.0,
            "stability_score_7d": 100.0
        }

    flapping_events = detect_flapping_events(recent_histories, flapping_threshold_min=1.0)
    flapping_count = len(flapping_events)
    flapping_intensity = flapping_count / (7 * 24)  

    change_intervals = []
    for i in range(1, len(recent_histories)):
        if recent_histories[i]['status'] != recent_histories[i-1]['status']:
            dt = (recent_histories[i]['createdAt'] - recent_histories[i-1]['createdAt']).total_seconds() / 60.0
            change_intervals.append(dt)

    avg_flap_duration = np.mean(change_intervals) if change_intervals else 999

    stability_score = 100.0

    if flapping_count > 0:
        stability_score *= max(0.01, 1 - (flapping_intensity / 5.0))

    if avg_flap_duration < 60:  
        stability_score *= max(0.05, avg_flap_duration / 60.0)

    if flapping_intensity > 2.0:
        stability_score *= 0.5  

    stability_score = max(0.0, min(100.0, stability_score))

    return {
        "flapping_events_7d": flapping_count,
        "flapping_intensity_7d": round(flapping_intensity, 3),
        "avg_flap_duration_7d_min": round(avg_flap_duration, 2),
        "stability_score_7d": round(stability_score, 2)
    }
    
def calculate_flapping_metrics(device_histories, snapshot_time, lookback_hours=24):
    lookback_start = snapshot_time - timedelta(hours=lookback_hours)
    recent_histories = [h for h in device_histories if lookback_start <= h['createdAt'] <= snapshot_time]

    if len(recent_histories) < 2:
        return {
            "flapping_events_24h": 0,
            "flapping_intensity_24h": 0.0,
            "avg_flap_duration_min": 0.0,
            "stability_score_24h": 100.0
        }

    flapping_events = detect_flapping_events(recent_histories, flapping_threshold_min=1.0)
    flapping_count = len(flapping_events)
    flapping_intensity = flapping_count / lookback_hours

    change_intervals = []
    for i in range(1, len(recent_histories)):
        if recent_histories[i]['status'] != recent_histories[i-1]['status']:
            dt = (recent_histories[i]['createdAt'] - recent_histories[i-1]['createdAt']).total_seconds() / 60.0
            change_intervals.append(dt)

    avg_flap_duration = np.mean(change_intervals) if change_intervals else 999

    stability_score = 100.0
    if flapping_count > 0:
        stability_score *= max(0.01, 1 - (flapping_intensity / 10)) 
    if avg_flap_duration < 30:
        stability_score *= max(0.1, avg_flap_duration / 30)

    return {
        "flapping_events_24h": flapping_count,
        "flapping_intensity_24h": round(flapping_intensity, 3),
        "avg_flap_duration_min": round(avg_flap_duration, 2),
        "stability_score_24h": round(max(0, min(100, stability_score)), 2)
    }
    
def label_failure_in_next_24h(row, segments_df, histories_dict):
    device_id = row["device_id"]
    ts = row["timestamp"]
    future_end = ts + timedelta(hours=24)

    device_segs = segments_df[segments_df["device_id"] == device_id]
    future_segs = device_segs[
        (device_segs["start"] >= ts) & 
        (device_segs["start"] <= future_end)
    ]

    total_downtime_next_24h = future_segs["duration_min"].sum()
    num_downtime_events = len(future_segs)
    has_long_outage = (future_segs["duration_min"] > 30).any()
    avg_downtime_duration = future_segs["duration_min"].mean() if len(future_segs) > 0 else 0

    device_histories = histories_dict.get(device_id, [])
    future_histories = [h for h in device_histories if ts <= h['createdAt'] <= future_end]
    flapping_events = detect_flapping_events(future_histories, flapping_threshold_min=5.0)  # looser
    num_flaps = len(flapping_events)

    is_failure = (
        has_long_outage or
        total_downtime_next_24h > 90 or                      
        (num_downtime_events >= 5 and total_downtime_next_24h > 60) or
        num_flaps >= 15 or                                  
        (num_flaps >= 8 and avg_downtime_duration < 10)     
    )

    return 1 if is_failure else 0

async def generate_training_dataset():
    print("Fetching status history from DB...")
    histories = await fetch_all_status_histories()
    if not histories:
        raise ValueError("No status history found in DB")
    print(f"Loaded {len(histories)} status records")

    history_dicts = []
    for h in histories:
        history_dicts.append({
            'deviceId': h['deviceId'],
            'status': h['status'],
            'createdAt': h['createdAt'],
            'remarks': h['remarks'],
            'userId': h['userId'],
            'device_name': h['device_name'],
            'ipAddress': h['ipAddress'],
            'property_name': h['property_name'],
            'location': h['location']
        })

    device_groups = {}
    for h in history_dicts:
        did = h['deviceId']
        device_groups.setdefault(did, []).append(h)

    print("Building downtime segments and flapping events...")
    all_segments = []
    device_histories_full = {did: sorted(hlist, key=lambda x: x['createdAt']) for did, hlist in device_groups.items()}

    for did, hlist in device_groups.items():
        hlist_sorted = sorted(hlist, key=lambda x: x['createdAt'])
        segments_df, _ = build_downtime_segments(hlist_sorted)
        if not segments_df.empty:
            segments = segments_df.to_dict('records')
            for seg in segments:
                seg['device_id'] = did
            all_segments.extend(segments)

    segments_df = pd.DataFrame(all_segments)
    if segments_df.empty:
        print("No downtime segments found. Creating empty dataframe.")
        segments_df = pd.DataFrame(columns=["device_id", "start", "end", "duration_min"])

    print(f"Generated {len(segments_df)} downtime segments")

    snapshot_rows = []
    for did, hlist in device_histories_full.items():
        ts_list = [h['createdAt'] for h in hlist]
        if not ts_list:
            continue
        min_ts = min(ts_list)
        max_ts = max(ts_list)
        current = min_ts.replace(minute=0, second=0, microsecond=0)
        while current <= max_ts:
            if current.hour % 6 == 0:
                snapshot_rows.append({
                    "device_id": did,
                    "timestamp": current
                })
            current += timedelta(hours=1)

    snapshot_df = pd.DataFrame(snapshot_rows)
    if snapshot_df.empty:
        raise ValueError("No snapshot timestamps generated")

    print(f"Generated {len(snapshot_df)} snapshot points")

    feature_rows = []
    for _, snap in snapshot_df.iterrows():
        did = snap["device_id"]
        ts = snap["timestamp"]

        day_ago = ts - timedelta(days=1)
        week_ago = ts - timedelta(days=7)

        dev_segs = segments_df[segments_df["device_id"] == did]
        past_24h = dev_segs[
            (dev_segs["end"] >= day_ago) & (dev_segs["start"] <= ts)
        ]
        past_7d = dev_segs[
            (dev_segs["end"] >= week_ago) & (dev_segs["start"] <= ts)
        ]

        downtime_24h = past_24h["duration_min"].sum() if not past_24h.empty else 0.0
        downtime_7d = past_7d["duration_min"].sum() if not past_7d.empty else 0.0
        events_24h = len(past_24h)

        uptime_24h = max(0, min(100, (1440 - downtime_24h) / 1440 * 100))
        uptime_7d = max(0, min(100, (10080 - downtime_7d) / 10080 * 100))
        avg_downtime = downtime_24h / max(events_24h, 1) if events_24h > 0 else 0

        device_histories = device_histories_full.get(did, [])
        flapping_24h = calculate_flapping_metrics(device_histories, ts, lookback_hours=24)
        flapping_7d = calculate_flapping_metrics_7d(device_histories, ts)

        feature_rows.append({
            "device_id": did,
            "timestamp": ts,
            "uptime_24h_pct": round(uptime_24h, 2),
            "uptime_7d_pct": round(uptime_7d, 2),
            "total_downtime_24h_min": round(downtime_24h, 2),
            "total_downtime_7d_min": round(downtime_7d, 2),
            "downtime_events_24h": events_24h,
            "downtime_events_7d": len(past_7d),
            "avg_downtime_duration_24h_min": round(avg_downtime, 2), 
            "avg_downtime_duration_7d_min": round(downtime_7d / max(len(past_7d), 1), 2), 
            **flapping_24h,
            **flapping_7d,
        })

    feature_df = pd.DataFrame(feature_rows)
    print(f"Feature matrix built: {len(feature_df)} rows")

    print("Labeling failures...")
    def will_fail_in_next_24h(row):
        did = row["device_id"]
        ts = row["timestamp"]
        future_start = ts
        future_end = ts + timedelta(hours=24)

        dev_segs = segments_df[segments_df["device_id"] == did]
        future_segs = dev_segs[
            (dev_segs["start"] >= future_start) &
            (dev_segs["start"] <= future_end)
        ]
        long_outages = future_segs[future_segs["duration_min"] > 30]
        return 1 if len(long_outages) > 0 else 0

    feature_df["failure_in_next_24h"] = feature_df.apply(will_fail_in_next_24h, axis=1)

    pos = feature_df["failure_in_next_24h"].sum()
    print(f"Positive failures (label=1): {pos} / {len(feature_df)} ({pos/len(feature_df)*100:.2f}%)")

    if pos == 0:
        print("WARNING: No positive samples! Model will be dummy.")
    else:
        failed = feature_df[feature_df["failure_in_next_24h"] == 1]
        print(f"Avg flapping events (failed): {failed['flapping_events_24h'].mean():.1f}")
        print(f"Avg flapping events (all): {feature_df['flapping_events_24h'].mean():.1f}")

    return feature_df

def train_and_save_model(df):
    if df.empty:
        print("No data available for training. Creating dummy model...")
        model = lgb.LGBMClassifier(n_estimators=10)
        X_dummy = pd.DataFrame({col: [0] for col in FEATURE_COLS})
        model.fit(X_dummy, [0])
    else:
        X = df[FEATURE_COLS]
        y = df["failure_in_next_24h"]

        if y.sum() == 0:
            print("No positive samples found. Training a dummy model...")
            model = lgb.LGBMClassifier(n_estimators=10, learning_rate=0.01)
            model.fit(X, [0]*len(y))
        else:
            print(f"Training with {y.sum()} positive samples...")
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
                "seed": 42,
                "scale_pos_weight": (len(y) - y.sum()) / y.sum() if y.sum() > 0 else 10,
            }

            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
            )

            y_pred = model.predict(X_test)
            print("ROC AUC:", roc_auc_score(y_test, y_pred))
            print(classification_report(y_test, (y_pred > 0.5).astype(int)))

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURE_COLS
    }, "models/lgbm_failure_model_with_flapping_500k.pkl")
    print("New model saved: models/lgbm_failure_model_with_flapping.pkl")

if __name__ == "__main__":
    print("Starting real-data training...")
    df = asyncio.run(generate_training_dataset())
    train_and_save_model(df)
    print("Training complete!")