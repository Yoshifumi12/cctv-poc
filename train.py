import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import asyncio
from datetime import datetime, timedelta
import asyncpg
import os

FEATURE_COLS = [
    "uptime_24h_pct",
    "uptime_7d_pct",
    "total_downtime_24h_min",
    "total_downtime_7d_min",
    "downtime_events_24h",
    "avg_downtime_duration_min"
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

    return pd.DataFrame(segments)

def label_failure_in_next_24h(row, segments_df):
    device_segs = segments_df[segments_df["device_id"] == row["device_id"]]
    future_segs = device_segs[
        (device_segs["start"] >= row["timestamp"]) &
        (device_segs["start"] <= row["timestamp"] + timedelta(hours=24))
    ]
    return 1 if len(future_segs[future_segs["duration_min"] > 30]) > 0 else 0

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
        if did not in device_groups:
            device_groups[did] = []
        device_groups[did].append(h)

    rows = []
    print("Building downtime segments...")
    all_segments = []

    for did, hlist in device_groups.items():
        hlist = sorted(hlist, key=lambda x: x['createdAt'])
        segments = []
        offline_start = None
        for i, h in enumerate(hlist):
            if h['status'] == "OFFLINE" and offline_start is None:
                offline_start = h['createdAt']
            elif h['status'] == "ONLINE" and offline_start is not None:
                duration = (h['createdAt'] - offline_start).total_seconds() / 60.0
                if duration > 0:
                    segments.append({
                        "device_id": did,
                        "start": offline_start,
                        "end": h['createdAt'],
                        "duration_min": duration
                    })
                offline_start = None

            if h['createdAt'].hour % 6 == 0 and h['createdAt'].minute < 5:
                rows.append({
                    "device_id": did,
                    "timestamp": h['createdAt'],
                    "status": h['status']
                })

        if offline_start is not None:
            now = datetime.utcnow()
            duration = (now - offline_start).total_seconds() / 60.0
            if duration > 0:
                segments.append({
                    "device_id": did,
                    "start": offline_start,
                    "end": now,
                    "duration_min": duration
                })

        all_segments.extend(segments)

    segments_df = pd.DataFrame(all_segments)
    snapshot_df = pd.DataFrame(rows)

    if snapshot_df.empty:
        print("No snapshots generated, creating manual snapshots...")
        for did, hlist in device_groups.items():
            if hlist:
                first_date = min(h['createdAt'] for h in hlist)
                last_date = max(h['createdAt'] for h in hlist)
                current = first_date
                while current <= last_date:
                    if current.hour % 6 == 0:
                        rows.append({
                            "device_id": did,
                            "timestamp": current,
                            "status": "ONLINE" 
                        })
                    current += timedelta(hours=1)
        
        snapshot_df = pd.DataFrame(rows)

    if snapshot_df.empty:
        raise ValueError("No snapshot timestamps generated")

    print(f"Generated {len(snapshot_df)} snapshots, {len(segments_df)} downtime events")

    print("Computing rolling uptime stats...")
    feature_rows = []

    for _, snap in snapshot_df.iterrows():
        did = snap["device_id"]
        ts = snap["timestamp"]

        day_ago = ts - timedelta(days=1)
        week_ago = ts - timedelta(days=7)

        dev_segs = segments_df[segments_df["device_id"] == did]
        past_24h = dev_segs[(dev_segs["end"] >= day_ago) & (dev_segs["start"] <= ts)]
        past_7d = dev_segs[(dev_segs["end"] >= week_ago) & (dev_segs["start"] <= ts)]

        downtime_24h = past_24h["duration_min"].sum()
        downtime_7d = past_7d["duration_min"].sum()
        events_24h = len(past_24h)

        uptime_24h = max(0, min(100, (1440 - downtime_24h) / 1440 * 100))
        uptime_7d = max(0, min(100, (10080 - downtime_7d) / 10080 * 100))
        avg_downtime = downtime_24h / max(events_24h, 1) if events_24h > 0 else 0

        feature_rows.append({
            "device_id": did,
            "timestamp": ts,
            "uptime_24h_pct": round(uptime_24h, 2),
            "uptime_7d_pct": round(uptime_7d, 2),
            "total_downtime_24h_min": round(downtime_24h, 2),
            "total_downtime_7d_min": round(downtime_7d, 2),
            "downtime_events_24h": events_24h,
            "avg_downtime_duration_min": round(avg_downtime, 2),
        })

    feature_df = pd.DataFrame(feature_rows)
    print(f"Feature matrix: {len(feature_df)} rows")

    if not feature_df.empty:
        print("Labeling failures...")
        feature_df["failure_in_next_24h"] = feature_df.apply(
            lambda row: label_failure_in_next_24h(row, segments_df), axis=1
        )

        pos = feature_df["failure_in_next_24h"].sum()
        print(f"Positive failures (label=1): {pos} / {len(feature_df)} ({pos/len(feature_df)*100:.2f}%)")
    else:
        print("No features generated, creating empty dataframe with failure column")
        feature_df["failure_in_next_24h"] = 0

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
                "seed": 42
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
    }, "models/lgbm_failure_model.pkl")
    print("New model saved: models/lgbm_failure_model.pkl")

if __name__ == "__main__":
    print("Starting real-data training...")
    df = asyncio.run(generate_training_dataset())
    train_and_save_model(df)
    print("Training complete!")