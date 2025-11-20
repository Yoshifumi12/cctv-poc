import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import asyncio
from datetime import timedelta
import asyncpg
import os
from common.metrics import build_downtime_segments, calculate_flapping_metrics

FEATURE_COLS = [
    "uptime_24h_pct", "uptime_7d_pct", "total_downtime_24h_min", "total_downtime_7d_min",
    "downtime_events_24h", "downtime_events_7d", "avg_downtime_duration_24h_min",
    "avg_downtime_duration_7d_min", "flapping_events_24h", "flapping_intensity_24h",
    "avg_flap_duration_min", "stability_score_24h", "flapping_events_7d",
    "flapping_intensity_7d", "avg_flap_duration_7d_min", "stability_score_7d"
]

async def fetch_histories():
    conn = await asyncpg.connect(
        host='localhost', port=5432, user='postgres', database='arms'
    )
    try:
        rows = await conn.fetch("""
            SELECT dsh."deviceId",
            dsh.status,
            dsh."createdAt"
            FROM "DeviceStatusHistory" dsh
            ORDER BY dsh."createdAt" ASC
        """)
        return [dict(r) for r in rows]
    finally:
        await conn.close()

async def generate_training_dataset():
    print("Fetching data from database...")
    histories = await fetch_histories()
    if not histories:
        raise ValueError("No data")

    df = pd.DataFrame(histories)
    print(f"Loaded {len(df)} records for {df['deviceId'].nunique()} devices")
    
    device_groups = {did: group.sort_values('createdAt') for did, group in df.groupby('deviceId')}

    print("Building downtime segments...")
    all_segments = []
    for did, group in device_groups.items():
        device_history = group.to_dict('records')
        segs = build_downtime_segments(device_history)
        for s in segs:
            s['device_id'] = did
        all_segments.extend(segs)
    segments_df = pd.DataFrame(all_segments)
    print(f"Created {len(segments_df)} downtime segments")

    print("Generating snapshots...")
    snapshots = []
    for did, group in device_groups.items():
        ts = group['createdAt']
        start = ts.min().replace(minute=0, second=0, microsecond=0)
        end = ts.max()
        current = start
        while current <= end:
            if current.hour % 6 == 0:
                snapshots.append({"device_id": did, "timestamp": current})
            current += timedelta(hours=1)
    snapshot_df = pd.DataFrame(snapshots)
    print(f"Created {len(snapshot_df)} snapshots")

    print("Calculating features...")
    rows = []
    for i, (_, row) in enumerate(snapshot_df.iterrows()):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(snapshot_df)} snapshots...")
            
        did, ts = row['device_id'], row['timestamp']
        day_ago = ts - timedelta(days=1)
        week_ago = ts - timedelta(days=7)

        dev_segs = segments_df[segments_df['device_id'] == did]
        past_24h = dev_segs[(dev_segs['end'] >= day_ago) & (dev_segs['start'] <= ts)]
        past_7d = dev_segs[(dev_segs['end'] >= week_ago) & (dev_segs['start'] <= ts)]

        dt24 = past_24h['duration_min'].sum()
        dt7d = past_7d['duration_min'].sum()
        events24 = len(past_24h)
        events7d = len(past_7d)

        uptime24 = max(0, min(100, (1440 - dt24) / 1440 * 100))
        uptime7d = max(0, min(100, (10080 - dt7d) / 10080 * 100))

        flapping_24h = calculate_flapping_metrics(device_groups[did].to_dict('records'), ts, lookback_hours=24)
        flapping_7d = calculate_flapping_metrics(device_groups[did].to_dict('records'), ts, lookback_hours=7*24)
        
        rows.append({
            "device_id": did,
            "timestamp": ts,
            "uptime_24h_pct": round(uptime24, 2),
            "uptime_7d_pct": round(uptime7d, 2),
            "total_downtime_24h_min": round(dt24, 2),
            "total_downtime_7d_min": round(dt7d, 2),
            "downtime_events_24h": events24,
            "downtime_events_7d": events7d,
            "avg_downtime_duration_24h_min": round(dt24 / max(1, events24), 2),
            "avg_downtime_duration_7d_min": round(dt7d / max(1, events7d), 2),
            **flapping_24h,
            **flapping_7d
        })

    feature_df = pd.DataFrame(rows)

    print("Calculating target variable...")
    def will_fail(row):
        did, ts = row['device_id'], row['timestamp']
        future = segments_df[
            (segments_df['device_id'] == did) &
            (segments_df['start'] >= ts) &
            (segments_df['start'] <= ts + timedelta(hours=24))
        ]
        long_outage = (future['duration_min'] > 30).any()
        total_dt = future['duration_min'].sum()
        flapping_count = len(future)  
        high_flapping = flapping_count > 20

        return 1 if (long_outage or total_dt > 90 or high_flapping) else 0

    feature_df['failure_in_next_24h'] = feature_df.apply(will_fail, axis=1)
    return feature_df

def train_and_save_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df['failure_in_next_24h']

    print(f"Dataset shape: {df.shape}")
    print(f"Positive samples: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=42
        )
        
    X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
    print(f"\nTarget distribution:")
    print(f"Train: {y_train.sum()} positives ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"Val: {y_val.sum()} positives ({y_val.sum()/len(y_val)*100:.2f}%)")
    print(f"Test: {y_test.sum()} positives ({y_test.sum()/len(y_test)*100:.2f}%)")
        
    params = {
            "objective": "binary", 
            "metric": "auc", 
            "learning_rate": 0.05,
            "num_leaves": 31, 
            "verbose": -1, 
            "seed": 42,
            "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
        }
        
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        
    print("\nTraining model...")
    model = lgb.train(
            params, 
            train_set, 
            valid_sets=[val_set],
            num_boost_round=1000, 
            callbacks=[lgb.early_stopping(50, verbose=True)]
        )
        
    val_pred = model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_pred)
        
    test_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, test_pred)
        
    print(f"\nValidation AUC: {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
        
    print("\nValidation Classification Report:")
    print(classification_report(y_val, (val_pred > 0.5).astype(int)))
        
    print("Test Classification Report:")
    print(classification_report(y_test, (test_pred > 0.5).astype(int)))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS}, "models/lgbm_failure_model.pkl")
    print("\nModel saved to models/lgbm_failure_model.pkl")
        
    return {
            'model': model,
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

async def main():
    print("Starting model training pipeline...")
    df = await generate_training_dataset()
    print(f"Generated training dataset with {len(df)} samples")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Target distribution: {df['failure_in_next_24h'].value_counts().to_dict()}")
    
    print("\nTraining model...")
    result = train_and_save_model(df)
    print("Training completed!")
    return result

if __name__ == "__main__":
    result = asyncio.run(main())