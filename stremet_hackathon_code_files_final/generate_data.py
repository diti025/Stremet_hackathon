import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# -----------------------------
# SETTINGS
# -----------------------------
n_units = 250

process_flow = ["laser_cutting", "bending", "welding"]

machine_map = {
    "laser_cutting": ["L-01", "L-02", "L-03"],
    "bending": ["B-01", "B-02"],
    "welding": ["W-01", "W-02"]
}

material_types = [
    "stainless_steel_304",
    "galvanized_steel",
    "aluminum_5754"
]

machine_base = {
    "L-01": {"process_step": "laser_cutting", "speed": 20, "force_kN": np.nan, "temperature_C": 300, "pressure_bar": 13, "power_usage_kW": 17},
    "L-02": {"process_step": "laser_cutting", "speed": 21, "force_kN": np.nan, "temperature_C": 310, "pressure_bar": 14, "power_usage_kW": 18},
    "L-03": {"process_step": "laser_cutting", "speed": 22, "force_kN": np.nan, "temperature_C": 320, "pressure_bar": 14, "power_usage_kW": 18},
    "B-01": {"process_step": "bending", "speed": 10, "force_kN": 78, "temperature_C": 27, "pressure_bar": np.nan, "power_usage_kW": 7},
    "B-02": {"process_step": "bending", "speed": 11, "force_kN": 82, "temperature_C": 28, "pressure_bar": np.nan, "power_usage_kW": 7},
    "W-01": {"process_step": "welding", "speed": 0.60, "force_kN": np.nan, "temperature_C": 1450, "pressure_bar": 3.0, "power_usage_kW": 9.0},
    "W-02": {"process_step": "welding", "speed": 0.55, "force_kN": np.nan, "temperature_C": 1475, "pressure_bar": 3.2, "power_usage_kW": 9.5},
}


def normalize_series(s):
    s = pd.Series(s)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------
# QUOTATIONS
# -----------------------------
quote_rows = []
for i in range(20):
    quote_id = f"Q-{3000 + i}"
    quantity = np.random.choice([40, 60, 80, 120, 180, 250])
    estimated_time_hours = round(np.random.uniform(6, 32), 1)
    estimated_cost_eur = round(quantity * np.random.uniform(18, 28), 0)
    promised_delivery_dt = datetime(2026, 4, 1) + timedelta(
        days=np.random.randint(1, 10),
        hours=np.random.randint(8, 18)
    )
    quote_status = np.random.choice(["accepted", "open"], p=[0.75, 0.25])

    quote_rows.append({
        "quote_id": quote_id,
        "quantity": quantity,
        "estimated_time_hours": estimated_time_hours,
        "estimated_cost_eur": estimated_cost_eur,
        "promised_delivery": promised_delivery_dt.strftime("%Y-%m-%d %H:%M"),
        "quote_status": quote_status
    })

quotes_df = pd.DataFrame(quote_rows)

# -----------------------------
# MATERIAL INFORMATION
# -----------------------------
materials_rows = []
for i in range(n_units):
    unit_id = f"U-{1042 + i}"
    quote = quotes_df.sample(1, random_state=np.random.randint(0, 100000)).iloc[0]

    material_type = np.random.choice(material_types, p=[0.35, 0.35, 0.30])
    thickness_mm = np.random.choice([1.5, 2.0, 2.5, 3.0, 4.0], p=[0.2, 0.3, 0.2, 0.2, 0.1])
    weight_kg = round(np.random.uniform(1.0, 3.5), 2)
    batch_id = f"MB-{2200 + np.random.randint(1, 25)}"

    materials_rows.append({
        "unit_id": unit_id,
        "quote_id": quote["quote_id"],
        "material_type": material_type,
        "thickness_mm": thickness_mm,
        "batch_id": batch_id,
        "weight_kg": weight_kg
    })

materials_df = pd.DataFrame(materials_rows)

# -----------------------------
# PRODUCTION EVENTS
# -----------------------------
events_rows = []
machine_param_rows = []
queue_rows = []

base_day = datetime(2026, 3, 28, 8, 0, 0)

for i in range(n_units):
    unit_id = f"U-{1042 + i}"
    unit_material = materials_df.loc[materials_df["unit_id"] == unit_id].iloc[0]
    current_time = base_day + timedelta(minutes=np.random.randint(0, 180))

    for process_step in process_flow:
        machine_id = np.random.choice(machine_map[process_step])

        queue_length_units = np.random.randint(2, 12)
        average_wait_time_min = np.random.randint(8, 40)

        queue_rows.append({
            "unit_id": unit_id,
            "process_step": process_step,
            "machine_id": machine_id,
            "queue_length_units": queue_length_units,
            "average_wait_time_min": average_wait_time_min,
            "queue_timestamp": current_time.strftime("%H:%M")
        })

        base_params = machine_base[machine_id]
        speed = round(base_params["speed"] * np.random.uniform(0.9, 1.1), 2) if pd.notna(base_params["speed"]) else np.nan
        force_kN = round(base_params["force_kN"] * np.random.uniform(0.9, 1.1), 2) if pd.notna(base_params["force_kN"]) else np.nan
        temperature_C = round(base_params["temperature_C"] * np.random.uniform(0.95, 1.08), 2) if pd.notna(base_params["temperature_C"]) else np.nan
        pressure_bar = round(base_params["pressure_bar"] * np.random.uniform(0.9, 1.1), 2) if pd.notna(base_params["pressure_bar"]) else np.nan
        power_usage_kW = round(base_params["power_usage_kW"] * np.random.uniform(0.9, 1.15), 2) if pd.notna(base_params["power_usage_kW"]) else np.nan

        machine_param_rows.append({
            "unit_id": unit_id,
            "process_step": process_step,
            "machine_id": machine_id,
            "speed": speed,
            "force_kN": force_kN,
            "temperature_C": temperature_C,
            "pressure_bar": pressure_bar,
            "power_usage_kW": power_usage_kW
        })

        base_duration = {
            "laser_cutting": 4,
            "bending": 5,
            "welding": 7
        }[process_step]

        thickness_penalty = unit_material["thickness_mm"] * 0.8
        queue_penalty = queue_length_units * 0.4

        duration_min = max(
            2,
            round(np.random.normal(base_duration + thickness_penalty + queue_penalty / 3, 1.2), 2)
        )

        start_time = current_time + timedelta(minutes=average_wait_time_min)
        end_time = start_time + timedelta(minutes=duration_min)

        events_rows.append({
            "unit_id": unit_id,
            "process_step": process_step,
            "machine_id": machine_id,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed"
        })

        current_time = end_time + timedelta(minutes=np.random.randint(2, 12))

events_df = pd.DataFrame(events_rows)
machine_params_df = pd.DataFrame(machine_param_rows)
queue_df = pd.DataFrame(queue_rows)

# -----------------------------
# TIME FEATURES
# -----------------------------
events_df["start_time_dt"] = pd.to_datetime(events_df["start_time"])
events_df["end_time_dt"] = pd.to_datetime(events_df["end_time"])
events_df["time_taken_min"] = (
    (events_df["end_time_dt"] - events_df["start_time_dt"]).dt.total_seconds() / 60
)

# -----------------------------
# MERGE ALL TABLES
# -----------------------------
features_df = (
    events_df
    .merge(machine_params_df, on=["unit_id", "process_step", "machine_id"], how="left")
    .merge(queue_df, on=["unit_id", "process_step", "machine_id"], how="left")
    .merge(materials_df, on="unit_id", how="left")
    .merge(quotes_df, on="quote_id", how="left")
)

# -----------------------------
# RISK COMPONENTS
# -----------------------------
baseline_time = features_df.groupby("process_step")["time_taken_min"].transform("median")
features_df["time_ratio"] = features_df["time_taken_min"] / baseline_time

baseline_queue = features_df.groupby("process_step")["queue_length_units"].transform("median")
features_df["queue_ratio"] = features_df["queue_length_units"] / baseline_queue

baseline_wait = features_df.groupby("process_step")["average_wait_time_min"].transform("median")
features_df["wait_ratio"] = features_df["average_wait_time_min"] / baseline_wait

temp_baseline = features_df.groupby("process_step")["temperature_C"].transform("median")
features_df["temp_ratio"] = features_df["temperature_C"] / temp_baseline

power_baseline = features_df.groupby("process_step")["power_usage_kW"].transform("median")
features_df["power_ratio"] = features_df["power_usage_kW"] / power_baseline

features_df["material_difficulty"] = np.select(
    [
        features_df["material_type"] == "stainless_steel_304",
        features_df["material_type"] == "galvanized_steel",
        features_df["material_type"] == "aluminum_5754"
    ],
    [1.00, 0.75, 0.55],
    default=0.60
)

features_df["thickness_norm"] = normalize_series(features_df["thickness_mm"])

features_df["process_complexity"] = np.select(
    [
        features_df["process_step"] == "laser_cutting",
        features_df["process_step"] == "bending",
        features_df["process_step"] == "welding"
    ],
    [0.55, 0.70, 1.00],
    default=0.60
)

features_df["operational_stress"] = (
    features_df["time_ratio"] *
    features_df["queue_ratio"] *
    features_df["temp_ratio"].fillna(1) *
    features_df["power_ratio"].fillna(1)
)

features_df["risk_score_raw"] = (
    features_df["operational_stress"] *
    (1 + 0.5 * features_df["thickness_norm"]) *
    features_df["material_difficulty"] *
    features_df["process_complexity"]
)

risk_norm = normalize_series(features_df["risk_score_raw"])
features_df["formula_risk_probability"] = sigmoid((risk_norm - 0.55) * 7)
features_df["risk_probability"] = features_df["formula_risk_probability"]

features_df["defect_flag"] = np.random.binomial(1, features_df["formula_risk_probability"])

# -----------------------------
# QUALITY INSPECTION TABLE
# -----------------------------
inspection_rows = []

for _, row in features_df.iterrows():
    if row["defect_flag"] == 1:
        if row["process_step"] == "welding":
            defect_type = np.random.choice(
                ["surface_porosity", "weak_joint", "burn_through"],
                p=[0.5, 0.3, 0.2]
            )
        elif row["process_step"] == "bending":
            defect_type = np.random.choice(
                ["angle_deviation", "surface_crack", "springback"],
                p=[0.45, 0.25, 0.30]
            )
        else:
            defect_type = np.random.choice(
                ["edge_burr", "incomplete_cut", "heat_mark"],
                p=[0.5, 0.3, 0.2]
            )

        severity = np.random.choice(["low", "medium", "high"], p=[0.45, 0.4, 0.15])
        result = "fail"
    else:
        defect_type = "-"
        severity = "-"
        result = "pass"

    inspection_rows.append({
        "unit_id": row["unit_id"],
        "process_step": row["process_step"],
        "result": result,
        "defect_type": defect_type,
        "severity": severity,
        "inspection_timestamp": row["end_time_dt"].strftime("%Y-%m-%d %H:%M:%S")
    })

inspection_df = pd.DataFrame(inspection_rows)

# -----------------------------
# FINAL DATASET
# -----------------------------
final_df = features_df.merge(
    inspection_df,
    on=["unit_id", "process_step"],
    how="left"
)

final_df["storage_location"] = np.where(
    final_df["process_step"] == "welding",
    np.random.choice(["Zone-C", "Zone-D"], len(final_df)),
    np.random.choice(["Zone-A", "Zone-B"], len(final_df))
)

# QR / access / device fields
final_df["qr_code"] = "QR_" + final_df["unit_id"].astype(str)
final_df["recommended_device"] = np.where(
    final_df["process_step"].isin(["laser_cutting", "bending", "welding"]),
    "phone_or_ipad",
    "desktop"
)

final_df = final_df.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore")

# -----------------------------
# SAVE FILES
# -----------------------------
final_df.to_csv("production_data.csv", index=False)
inspection_df.to_csv("quality_inspection.csv", index=False)

print("Saved production_data.csv and quality_inspection.csv")
print(final_df.head())
