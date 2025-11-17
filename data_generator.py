import pandas as pd
from datetime import datetime, timedelta
import random
import uuid

def generate_fake_uptime_data(n_devices=50, days=30):
    data = []
    start_date = datetime.now() - timedelta(days=days)

    for _ in range(n_devices):
        device_id = str(uuid.uuid4())
        ip = f"192.168.1.{random.randint(10, 200)}"
        property_id = str(uuid.uuid4())

        current_time = start_date
        status = "ONLINE"
        downtime_start = None

        while current_time < datetime.now():
            duration = random.randint(60, 360)  
            next_time = current_time + timedelta(minutes=duration)

            if status == "ONLINE":
                if random.random() < 0.05:
                    status = "OFFLINE"
                    downtime_start = current_time
            else:
                if random.random() < 0.3 or (downtime_start and (current_time - downtime_start).total_seconds() > 3600):
                    status = "ONLINE"
                    downtime_dur = (current_time - downtime_start).total_seconds() / 60
                    if downtime_dur > 30:
                        data.append({
                            "device_id": device_id,
                            "ip_address": ip,
                            "property_id": property_id,
                            "timestamp": current_time,
                            "status": "OFFLINE",
                            "downtime_duration_min": round(downtime_dur, 2),
                            "failure_in_next_24h": 1
                        })
                    downtime_start = None

            if current_time.hour in [0, 6, 12, 18]:
                uptime_24h = round(random.uniform(90, 100) if status == "ONLINE" else random.uniform(50, 95), 2)
                uptime_7d = round(random.uniform(85, 99.9) if status == "ONLINE" else random.uniform(60, 90), 2)
                total_downtime_24h = round((1440 - uptime_24h * 14.4), 2) if uptime_24h < 100 else 0
                total_downtime_7d = round((10080 - uptime_7d * 100.8), 2) if uptime_7d < 100 else 0

                data.append({
                    "device_id": device_id,
                    "ip_address": ip,
                    "property_id": property_id,
                    "timestamp": current_time,
                    "status": status,
                    "uptime_24h_pct": uptime_24h,
                    "uptime_7d_pct": uptime_7d,
                    "total_downtime_24h_min": total_downtime_24h,
                    "total_downtime_7d_min": total_downtime_7d,
                    "downtime_events_24h": random.randint(0, 3),
                    "avg_downtime_duration_min": round(random.uniform(5, 120), 2),
                    "failure_in_next_24h": 0  
                })

            current_time = next_time

    df = pd.DataFrame(data)
    
    df = df.sort_values(["device_id", "timestamp"])
    df["failure_in_next_24h"] = 0

    for i, row in df.iterrows():
        device_mask = df["device_id"] == row["device_id"]
        future_mask = df["timestamp"] > row["timestamp"]
        window_mask = df["timestamp"] <= row["timestamp"] + timedelta(hours=24)
        future_downtime = df[device_mask & future_mask & window_mask]
        if len(future_downtime[future_downtime["status"] == "OFFLINE"]) > 0:
            if future_downtime["downtime_duration_min"].max() > 30:
                df.at[i, "failure_in_next_24h"] = 1

    return df

if __name__ == "__main__":
    df = generate_fake_uptime_data()
    df.to_csv("fake_data.csv", index=False)
    print(f"Generated {len(df)} rows → fake_data.csv")