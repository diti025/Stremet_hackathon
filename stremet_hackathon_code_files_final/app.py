import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import json
REPORTS_FILE = "employee_reports.csv"

def load_reports():
    if os.path.exists(REPORTS_FILE):
        return pd.read_csv(REPORTS_FILE)
    else:
        return pd.DataFrame(columns=[
            "qr_code", "unit_id", "machine_id",
            "start_time", "end_time",
            "actual_temperature", "actual_pressure", "actual_power",
            "ideal_temperature", "ideal_pressure", "ideal_power",
            "issue_type", "notes", "timestamp"
        ])

def save_report(row):
    df = load_reports()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(REPORTS_FILE, index=False)

st.set_page_config(page_title="Stremet Executive Operations Dashboard", layout="wide")
# -------------------------
# AUTH FUNCTIONS
# -------------------------
def load_users():
    with open("users.json", "r") as f:
        return json.load(f)

def authenticate(username, password):
    users = load_users()
    for user in users.values():
        if user["username"] == username and user["password"] == password:
            return user
    return None

# -------------------------
# SESSION STATE INIT
# -------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -------------------------
# LOGIN SCREEN
# -------------------------
if st.session_state.user is None:
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = authenticate(username, password)

        if user:
            st.session_state.user = user
            st.success(f"Welcome {user['role']}")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -------------------------
# AFTER LOGIN
# -------------------------
# Logout button with rerun
if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# Get user from session state - this will never be None here because we stopped above if it was None
user = st.session_state.user
if user is None:
    st.session_state.user = None
    st.error("Session error. Please login again.")
    st.stop()

role = user.get("role") if isinstance(user, dict) else None

if role is None:
    st.session_state.user = None
    st.error("Invalid session. Please login again.")
    st.stop()
# ==================================================
# EMPLOYEE VIEW (SIMPLE)
# ==================================================
# --------------------------------------------------
# EMPLOYEE VIEW
# --------------------------------------------------
if role == "employee":
    st.title("Employee Welding Interface")

    data = pd.read_csv("production_data.csv")

    st.subheader("🔍 Scan QR Code")
    welding_units = data[data["process_step"] == "welding"]

    qr_input = st.selectbox(
        "Select Welding Unit",
        welding_units["qr_code"].unique()
    )
    if qr_input:
        # ✅ Filter ONLY welding stage
        unit_row = data[data["qr_code"] == qr_input]

        if unit_row.empty:
            st.error("Invalid QR Code")
            st.stop()

# ✅ Just pick one row (no filtering)
        row = unit_row.iloc[0]

        st.success(f"Unit: {row['unit_id']} | Machine: {row['machine_id']}")

        # --------------------------------------------------
        # REPORT FORM
        # --------------------------------------------------
        st.subheader("📝 Welding Report")

        start_time = st.time_input("Start Time")
        end_time = st.time_input("End Time")

        st.markdown("### Actual Parameters")
        actual_temp = st.number_input("Temperature (°C)", value=float(row["temperature_C"]))
        actual_pressure = st.number_input("Pressure (bar)", value=float(row["pressure_bar"]))
        actual_power = st.number_input("Power (kW)", value=float(row["power_usage_kW"]))

        st.markdown("### Ideal Parameters (Suggested)")
        ideal_temp = st.number_input("Ideal Temperature", value=float(row["temperature_C"]))
        ideal_pressure = st.number_input("Ideal Pressure", value=float(row["pressure_bar"]))
        ideal_power = st.number_input("Ideal Power", value=float(row["power_usage_kW"]))

        issue_type = st.selectbox("Issue Type", [
            "none",
            "surface_porosity",
            "weak_joint",
            "burn_through",
            "machine_instability"
        ])

        notes = st.text_area("Notes")

        if st.button("Submit Report"):
            report = {
                "qr_code": qr_input,
                "unit_id": row["unit_id"],
                "machine_id": row["machine_id"],
                "start_time": str(start_time),
                "end_time": str(end_time),
                "actual_temperature": actual_temp,
                "actual_pressure": actual_pressure,
                "actual_power": actual_power,
                "ideal_temperature": ideal_temp,
                "ideal_pressure": ideal_pressure,
                "ideal_power": ideal_power,
                "issue_type": issue_type,
                "notes": notes,
                "timestamp": pd.Timestamp.now()
            }

            save_report(report)
            st.success("Report submitted successfully ✅")

    st.stop()

elif role == "manager":

    # --------------------------------------------------
    # PROFESSIONAL STYLING
    # --------------------------------------------------
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f5f7fa;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------
    data = pd.read_csv("production_data.csv")
    data["process_step"] = data["process_step"].str.strip().str.lower()
    data["qr_code"] = data["qr_code"].str.strip()
    model = joblib.load("defect_model.pkl")
    model_columns = joblib.load("model_columns.pkl")

    if os.path.exists("model_coefficients.csv"):
        coef_df = pd.read_csv("model_coefficients.csv")
    else:
        coef_df = pd.DataFrame({"feature": [], "coefficient": []})

    # --------------------------------------------------
    # SAFE REQUIRED COLUMNS
    # --------------------------------------------------
    required_cols = {
        "unit_id": "",
        "qr_code": "",
        "quote_id": "",
        "process_step": "",
        "machine_id": "",
        "result": "pass",
        "defect_type": "-",
        "severity": "-",
        "time_taken_min": 0.0,
        "queue_length_units": 0.0,
        "average_wait_time_min": 0.0,
        "temperature_C": np.nan,
        "pressure_bar": np.nan,
        "power_usage_kW": np.nan,
        "material_type": "",
        "thickness_mm": np.nan,
        "weight_kg": 0.0,
        "batch_id": "",
        "estimated_time_hours": np.nan,
        "estimated_cost_eur": np.nan,
        "risk_score_raw": 0.0,
        "formula_risk_probability": 0.0,
        "quantity": 1.0
    }
    for col, default in required_cols.items():
        if col not in data.columns:
            data[col] = default

    numeric_cols_all = [
        "time_taken_min", "queue_length_units", "average_wait_time_min",
        "temperature_C", "pressure_bar", "power_usage_kW", "thickness_mm",
        "weight_kg", "estimated_time_hours", "estimated_cost_eur",
        "risk_score_raw", "formula_risk_probability", "quantity"
    ]
    for col in numeric_cols_all:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["quantity"] = data["quantity"].fillna(1).replace(0, 1)
    data["weight_kg"] = data["weight_kg"].fillna(0)
    data["time_taken_min"] = data["time_taken_min"].fillna(0)
    data["queue_length_units"] = data["queue_length_units"].fillna(0)
    data["average_wait_time_min"] = data["average_wait_time_min"].fillna(0)
    data["risk_score_raw"] = data["risk_score_raw"].fillna(0)
    data["formula_risk_probability"] = data["formula_risk_probability"].fillna(0)
    data["result"] = data["result"].fillna("pass")
    data["is_fail"] = (data["result"] == "fail").astype(int)

    process_order = {
        "laser_cutting": 1,
        "bending": 2,
        "welding": 3
    }
    data["process_rank"] = data["process_step"].map(process_order).fillna(999)

    # --------------------------------------------------
    # FIXED COST MODEL
    # --------------------------------------------------
    LABOR_RATE_EUR_PER_HOUR = 60.0
    LABOR_RATE_EUR_PER_MIN = LABOR_RATE_EUR_PER_HOUR / 60.0
    SETUP_COST_PER_ORDER_EUR = 50.0

    def material_rate(material_type: str) -> float:
        if pd.isna(material_type):
            return 0.95
        m = str(material_type).lower()
        if "aluminum" in m:
            return 4.0
        if "stainless" in m:
            return 2.8
        if "galvanized" in m or "zinc" in m:
            return 1.1
        if "steel" in m:
            return 0.95
        return 0.95

    data["material_rate_eur_per_kg"] = data["material_type"].apply(material_rate)
    data["material_cost_eur"] = data["weight_kg"] * data["material_rate_eur_per_kg"]
    data["processing_cost_eur"] = data["time_taken_min"] * LABOR_RATE_EUR_PER_MIN
    data["delay_cost_eur"] = data["average_wait_time_min"] * LABOR_RATE_EUR_PER_MIN
    data["setup_share_eur"] = SETUP_COST_PER_ORDER_EUR / data["quantity"]
    data["event_value_eur"] = data["material_cost_eur"] + data["processing_cost_eur"] + data["setup_share_eur"]

    # --------------------------------------------------
    # FILTERS
    # --------------------------------------------------
    st.sidebar.header("Filters")

    available_steps = sorted([x for x in data["process_step"].dropna().unique() if x != ""])
    available_machines = sorted([x for x in data["machine_id"].dropna().unique() if x != ""])
    available_materials = sorted([x for x in data["material_type"].dropna().unique() if x != ""])

    selected_steps = st.sidebar.multiselect("Process step", available_steps, default=available_steps)
    selected_machines = st.sidebar.multiselect("Machine", available_machines, default=available_machines)
    selected_materials = st.sidebar.multiselect("Material", available_materials, default=available_materials)

    filtered = data.copy()
    if selected_steps:
        filtered = filtered[filtered["process_step"].isin(selected_steps)]
    if selected_machines:
        filtered = filtered[filtered["machine_id"].isin(selected_machines)]
    if selected_materials:
        filtered = filtered[filtered["material_type"].isin(selected_materials)]

    # --------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------
    def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
        base = df[[
            "process_step",
            "machine_id",
            "time_taken_min",
            "queue_length_units",
            "average_wait_time_min",
            "temperature_C",
            "pressure_bar",
            "power_usage_kW",
            "material_type",
            "thickness_mm",
            "estimated_time_hours",
            "estimated_cost_eur",
            "risk_score_raw"
        ]].copy()

        numeric_cols = [
            "time_taken_min",
            "queue_length_units",
            "average_wait_time_min",
            "temperature_C",
            "pressure_bar",
            "power_usage_kW",
            "thickness_mm",
            "estimated_time_hours",
            "estimated_cost_eur",
            "risk_score_raw"
        ]

        for col in numeric_cols:
            base[col] = pd.to_numeric(base[col], errors="coerce")
            fill_value = pd.to_numeric(filtered[col], errors="coerce").median() if col in filtered.columns else 0
            if pd.isna(fill_value):
                fill_value = 0
            base[col] = base[col].fillna(fill_value)

        base["process_step"] = base["process_step"].fillna("unknown")
        base["machine_id"] = base["machine_id"].fillna("unknown")
        base["material_type"] = base["material_type"].fillna("unknown")

        encoded = pd.get_dummies(
            base,
            columns=["process_step", "machine_id", "material_type"],
            drop_first=False
        )

        for col in model_columns:
            if col not in encoded.columns:
                encoded[col] = 0

        return encoded[model_columns].fillna(0)

    def risk_label(prob: float) -> str:
        if prob >= 0.60:
            return "High"
        if prob >= 0.35:
            return "Medium"
        return "Low"

    # safe event-level prediction
    try:
        X_all = build_model_input(filtered)
        filtered["predicted_failure_risk"] = model.predict_proba(X_all)[:, 1]
    except Exception:
        filtered["predicted_failure_risk"] = filtered["formula_risk_probability"].fillna(0)

    filtered["risk_level"] = filtered["predicted_failure_risk"].apply(risk_label)

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def readable_feature_name(feature: str) -> str:
        mapping = {
            "time_taken_min": "Long processing time",
            "queue_length_units": "High queue length",
            "average_wait_time_min": "Long waiting time",
            "temperature_C": "Temperature deviation",
            "pressure_bar": "Pressure deviation",
            "power_usage_kW": "High power usage",
            "thickness_mm": "High material thickness",
            "estimated_time_hours": "Large quoted production effort",
            "estimated_cost_eur": "High order complexity",
            "risk_score_raw": "Elevated combined process stress",
            "process_step_laser_cutting": "Laser cutting conditions",
            "process_step_bending": "Bending process complexity",
            "process_step_welding": "Welding process complexity",
            "machine_id_L-01": "Machine L-01 behavior",
            "machine_id_L-02": "Machine L-02 behavior",
            "machine_id_L-03": "Machine L-03 behavior",
            "machine_id_B-01": "Machine B-01 behavior",
            "machine_id_B-02": "Machine B-02 behavior",
            "machine_id_W-01": "Machine W-01 behavior",
            "machine_id_W-02": "Machine W-02 behavior",
            "material_type_stainless_steel_304": "Stainless steel batch complexity",
            "material_type_galvanized_steel": "Galvanized steel handling",
            "material_type_aluminum_5754": "Aluminum handling"
        }
        return mapping.get(feature, feature)

    def get_driver_details(row_df: pd.DataFrame) -> tuple[list[str], list[str]]:
        try:
            input_encoded = build_model_input(row_df)
            scaler_ct = model.named_steps["scaler"]
            lr_model = model.named_steps["model"]

            numeric_cols = list(scaler_ct.transformers_[0][2])
            remainder_cols = [c for c in input_encoded.columns if c not in numeric_cols]
            transformed = scaler_ct.transform(input_encoded)
            feature_names = numeric_cols + remainder_cols

            contributions = pd.Series(
                transformed[0] * lr_model.coef_[0],
                index=feature_names
            ).sort_values(ascending=False)

            positive = contributions[contributions > 0].head(3)
            negative = contributions[contributions < 0].tail(3)

            risk_drivers = [readable_feature_name(f) for f in positive.index.tolist()]
            protective_drivers = [readable_feature_name(f) for f in negative.index.tolist()]

            if not risk_drivers:
                risk_drivers = ["Operational stress", "Queue pressure", "Process complexity"]
            if not protective_drivers:
                protective_drivers = ["Stable batch behavior", "Normal loading", "Consistent machine performance"]

            return risk_drivers, protective_drivers

        except Exception:
            return (
                ["Operational stress", "Queue pressure", "Process complexity"],
                ["Stable batch behavior", "Normal loading", "Consistent machine performance"]
            )

    def recommendation_from_causes(causes: list[str], process_step: str) -> str:
        text = " | ".join(causes).lower()
        if "queue" in text or "wait" in text:
            return "Reduce queue load on this process, resequence orders, or reroute units to a lower-load machine."
        if "machine" in text:
            return "Inspect calibration and maintenance history, then compare settings with the best-performing machine in the same process."
        if "temperature" in text or "pressure" in text or "power" in text:
            return "Retune process parameters to process-standard settings and strengthen parameter monitoring at this step."
        if "material" in text or "thickness" in text or "stainless" in text or "batch" in text:
            return "Use batch-specific settings, flag this batch for extra inspection, and standardize settings for similar thickness/material combinations."
        if "welding" in text:
            return "Add a pre-weld checkpoint, tighten welding settings, and prioritize experienced setup for this product type."
        return f"Inspect before the next {process_step} step and standardize settings around the main risk drivers."

    def estimate_unit_costs(events_df: pd.DataFrame) -> pd.DataFrame:
        latest = events_df.sort_values(["unit_id", "process_rank"]).groupby("unit_id", as_index=False).tail(1).copy()
        total_time = events_df.groupby("unit_id")["time_taken_min"].sum().reset_index(name="total_time_min")
        latest = latest.merge(total_time, on="unit_id", how="left")

        latest["material_rate_eur_per_kg"] = latest["material_type"].apply(material_rate)
        latest["material_cost_eur"] = latest["weight_kg"] * latest["material_rate_eur_per_kg"]
        latest["labor_cost_eur"] = (latest["total_time_min"] / 60.0) * LABOR_RATE_EUR_PER_HOUR
        latest["setup_share_eur"] = SETUP_COST_PER_ORDER_EUR / latest["quantity"].replace(0, 1)
        latest["unit_total_cost_eur"] = latest["material_cost_eur"] + latest["labor_cost_eur"] + latest["setup_share_eur"]
        return latest

    def estimate_business_impact(row: pd.Series, process_wait_medians: dict) -> tuple[float, float]:
        prob = float(row.get("predicted_failure_risk", 0))
        expected_scrap_cost = prob * float(row.get("unit_total_cost_eur", 0))
        process_median_wait = process_wait_medians.get(row.get("process_step", ""), 0)
        excess_wait_min = max(0, float(row.get("average_wait_time_min", 0)) - process_median_wait)
        expected_delay_cost = excess_wait_min * LABOR_RATE_EUR_PER_MIN
        total_impact = expected_scrap_cost + expected_delay_cost
        return total_impact, excess_wait_min

    def solution_for_scrap(scrap_rate_value: float, top_scrap_process: str) -> str:
        if scrap_rate_value >= 15:
            return f"Scrap is materially high. Start immediate review in {top_scrap_process}, inspect high-risk units before the next step, and standardize settings for the worst-performing machines."
        if scrap_rate_value >= 7:
            return f"Scrap is moderate. Prioritize targeted interventions in {top_scrap_process}, especially on high-risk units and problematic material batches."
        return f"Scrap is relatively controlled. Maintain current controls and preserve the best-performing settings in {top_scrap_process}."

    def solution_for_efficiency(efficiency_value: float) -> str:
        if efficiency_value < 85:
            return "Production efficiency is under pressure. Focus on reducing scrap, shortening machine wait times, and intervening earlier on high-risk units."
        if efficiency_value < 93:
            return "Production efficiency is acceptable but can improve. Prioritize bottleneck relief and batch-specific quality controls."
        return "Production efficiency is strong. Preserve current working conditions and standardize best-performing machine settings."

    def solution_for_bottleneck(machine: str, process: str) -> str:
        return (
            f"Relieve {machine} by redistributing load within {process}, tightening schedule sequencing, "
            f"and reviewing maintenance or setup calibration to cut waiting time."
        )

    # --------------------------------------------------
    # UNIT-LEVEL EXECUTIVE TABLE
    # --------------------------------------------------
    latest_unit_rows = estimate_unit_costs(filtered)

    unit_fail_summary = (
        filtered.groupby("unit_id")["is_fail"]
        .max()
        .reset_index()
        .rename(columns={"is_fail": "unit_failed"})
    )

    latest_event_info = (
        filtered.sort_values(["unit_id", "process_rank"])
        .groupby("unit_id", as_index=False)
        .tail(1)[[
            "unit_id",
            "predicted_failure_risk",
            "risk_level",
            "formula_risk_probability",
            "process_step",
            "machine_id",
            "batch_id",
            "average_wait_time_min",
            "qr_code"
        ]]
        .copy()
    )

    latest_unit_rows = latest_unit_rows.merge(unit_fail_summary, on="unit_id", how="left")
    latest_unit_rows = latest_unit_rows.merge(
        latest_event_info.drop_duplicates(subset=["unit_id"]),
        on="unit_id",
        how="left",
        suffixes=("", "_event")
    )

    # final safety
    if "predicted_failure_risk" not in latest_unit_rows.columns:
        latest_unit_rows["predicted_failure_risk"] = 0.0
    else:
        latest_unit_rows["predicted_failure_risk"] = pd.to_numeric(
            latest_unit_rows["predicted_failure_risk"], errors="coerce"
        ).fillna(0)

    if "risk_level" not in latest_unit_rows.columns:
        latest_unit_rows["risk_level"] = latest_unit_rows["predicted_failure_risk"].apply(risk_label)
    else:
        latest_unit_rows["risk_level"] = latest_unit_rows["risk_level"].fillna(
            latest_unit_rows["predicted_failure_risk"].apply(risk_label)
        )

    # --------------------------------------------------
    # BUSINESS METRICS
    # --------------------------------------------------
    total_units = latest_unit_rows["unit_id"].nunique()
    failed_units = int(latest_unit_rows["unit_failed"].fillna(0).sum()) if total_units > 0 else 0
    good_units = total_units - failed_units

    scrap_rate = (failed_units / total_units * 100) if total_units > 0 else 0
    direct_scrap_loss = latest_unit_rows.loc[latest_unit_rows["unit_failed"] == 1, "unit_total_cost_eur"].sum()
    production_efficiency = (good_units / total_units * 100) if total_units > 0 else 0

    high_risk_units = latest_unit_rows[latest_unit_rows["risk_level"] == "High"].copy()

    process_wait_medians = filtered.groupby("process_step")["average_wait_time_min"].median().to_dict()

    if not high_risk_units.empty:
        risk_driver_list = []
        protective_driver_list = []
        recommended_actions = []
        impact_costs = []
        delay_mins = []

        for _, row in high_risk_units.iterrows():
            row_df = filtered[filtered["unit_id"] == row["unit_id"]].sort_values("process_rank").tail(1)
            risk_drivers, protective_drivers = get_driver_details(row_df)
            action = recommendation_from_causes(risk_drivers, row.get("process_step", "production"))
            impact_cost, delay_min = estimate_business_impact(row, process_wait_medians)

            risk_driver_list.append(" | ".join(risk_drivers))
            protective_driver_list.append(" | ".join(protective_drivers))
            recommended_actions.append(action)
            impact_costs.append(round(impact_cost, 2))
            delay_mins.append(round(delay_min, 1))

        high_risk_units["top_3_risk_drivers"] = risk_driver_list
        high_risk_units["protective_drivers_to_preserve"] = protective_driver_list
        high_risk_units["suggested_solution"] = recommended_actions
        high_risk_units["estimated_business_impact_eur"] = impact_costs
        high_risk_units["estimated_delay_min"] = delay_mins

    high_risk_units = high_risk_units.sort_values("predicted_failure_risk", ascending=False).copy()
    preventable_loss = high_risk_units["estimated_business_impact_eur"].sum() if not high_risk_units.empty else 0

    # --------------------------------------------------
    # DELIVERY RELIABILITY
    # --------------------------------------------------
    if "quote_id" in filtered.columns and filtered["quote_id"].notna().any():
        quote_summary = (
            filtered.groupby("quote_id")
            .agg(
                actual_hours=("time_taken_min", lambda x: x.sum() / 60.0),
                delay_hours=("average_wait_time_min", lambda x: x.sum() / 60.0),
                quoted_hours=("estimated_time_hours", "first")
            )
            .reset_index()
        )
        quote_summary["total_hours"] = quote_summary["actual_hours"] + quote_summary["delay_hours"]
        quote_summary["quoted_hours"] = pd.to_numeric(quote_summary["quoted_hours"], errors="coerce")
        quote_summary["on_time"] = quote_summary["total_hours"] <= quote_summary["quoted_hours"]
        delivery_reliability = quote_summary["on_time"].mean() * 100 if len(quote_summary) > 0 else 100.0
    else:
        quote_summary = pd.DataFrame()
        delivery_reliability = 100.0

    # --------------------------------------------------
    # BOTTLENECK ANALYSIS
    # --------------------------------------------------
    machine_wait = (
        filtered.groupby("machine_id")["average_wait_time_min"]
        .mean()
        .sort_values(ascending=False)
    )

    overall_avg_wait = filtered["average_wait_time_min"].mean()

    if not machine_wait.empty:
        bottleneck_machine = machine_wait.idxmax()
        bottleneck_wait = machine_wait.max()

        bottleneck_machine_rows = filtered[filtered["machine_id"] == bottleneck_machine]
        slowing_process = (
            bottleneck_machine_rows.groupby("process_step")["average_wait_time_min"]
            .mean()
            .sort_values(ascending=False)
            .idxmax()
        )

        delay_pct = ((bottleneck_wait / overall_avg_wait) - 1) * 100 if overall_avg_wait > 0 else 0
        excess_wait_min = max(0, bottleneck_wait - overall_avg_wait) * len(bottleneck_machine_rows)
        bottleneck_delay_cost = excess_wait_min * LABOR_RATE_EUR_PER_MIN
    else:
        bottleneck_machine = "N/A"
        bottleneck_wait = 0
        slowing_process = "N/A"
        delay_pct = 0
        bottleneck_delay_cost = 0

    # --------------------------------------------------
    # BATCH + PARAMETER SUMMARIES
    # --------------------------------------------------
    batch_summary = (
        filtered.groupby(["batch_id", "material_type"], dropna=False)
        .agg(
            units=("unit_id", "nunique"),
            fail_rate=("is_fail", "mean"),
            avg_risk=("predicted_failure_risk", "mean"),
            avg_queue=("queue_length_units", "mean"),
            avg_time=("time_taken_min", "mean")
        )
        .reset_index()
    )

    if not batch_summary.empty:
        batch_summary["fail_rate"] = batch_summary["fail_rate"] * 100
        batch_summary = batch_summary.sort_values(["fail_rate", "avg_risk"], ascending=False)

    param_cols = ["time_taken_min", "queue_length_units", "average_wait_time_min", "temperature_C", "power_usage_kW"]
    param_compare = filtered.groupby("result")[param_cols].mean().reset_index()

    # --------------------------------------------------
    # IMPROVEMENT SCENARIO
    # --------------------------------------------------
    st.sidebar.header("Machine Improvement Scenario")

    available_machines = sorted([x for x in filtered["machine_id"].dropna().unique() if x != ""])
    scenario_machine = st.sidebar.selectbox(
        "Select machine to improve",
        available_machines if available_machines else ["N/A"]
    )

    scenario_action = st.sidebar.selectbox(
        "Select operational improvement",
        [
            "Reduce wait time to process median",
            "Reduce processing time to process median",
            "Retune process parameters to process median",
            "Full machine improvement package"
        ]
    )

    scenario_df = filtered.copy()

    machine_process = scenario_df.loc[scenario_df["machine_id"] == scenario_machine, "process_step"]
    machine_process = machine_process.mode().iloc[0] if len(machine_process) > 0 else None

    if machine_process is not None:
        process_rows = scenario_df["process_step"] == machine_process
        machine_rows = scenario_df["machine_id"] == scenario_machine

        process_wait_median = scenario_df.loc[process_rows, "average_wait_time_min"].median()
        process_time_median = scenario_df.loc[process_rows, "time_taken_min"].median()
        process_temp_median = scenario_df.loc[process_rows, "temperature_C"].median()
        process_pressure_median = scenario_df.loc[process_rows, "pressure_bar"].median()
        process_power_median = scenario_df.loc[process_rows, "power_usage_kW"].median()

        if scenario_action == "Reduce wait time to process median":
            scenario_df.loc[machine_rows, "average_wait_time_min"] = process_wait_median
        elif scenario_action == "Reduce processing time to process median":
            scenario_df.loc[machine_rows, "time_taken_min"] = process_time_median
        elif scenario_action == "Retune process parameters to process median":
            scenario_df.loc[machine_rows, "temperature_C"] = process_temp_median
            scenario_df.loc[machine_rows, "pressure_bar"] = process_pressure_median
            scenario_df.loc[machine_rows, "power_usage_kW"] = process_power_median
        elif scenario_action == "Full machine improvement package":
            scenario_df.loc[machine_rows, "average_wait_time_min"] = process_wait_median
            scenario_df.loc[machine_rows, "time_taken_min"] = process_time_median
            scenario_df.loc[machine_rows, "temperature_C"] = process_temp_median
            scenario_df.loc[machine_rows, "pressure_bar"] = process_pressure_median
            scenario_df.loc[machine_rows, "power_usage_kW"] = process_power_median

    try:
        scenario_X = build_model_input(scenario_df)
        scenario_df["predicted_failure_risk"] = model.predict_proba(scenario_X)[:, 1]
    except Exception:
        scenario_df["predicted_failure_risk"] = scenario_df["formula_risk_probability"].fillna(0)

    scenario_latest = (
        scenario_df.sort_values(["unit_id", "process_rank"])
        .groupby("unit_id", as_index=False)
        .tail(1)[["unit_id", "predicted_failure_risk"]]
        .copy()
    )

    if "predicted_failure_risk" not in scenario_latest.columns:
        scenario_latest["predicted_failure_risk"] = 0.0
    else:
        scenario_latest["predicted_failure_risk"] = pd.to_numeric(
            scenario_latest["predicted_failure_risk"], errors="coerce"
        ).fillna(0)

    scenario_high_risk_count = int((scenario_latest["predicted_failure_risk"] >= 0.60).sum())
    baseline_high_risk_count = int((latest_unit_rows["predicted_failure_risk"] >= 0.60).sum())

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------
    st.title("Stremet Executive Operations Dashboard")
    st.caption(
        "Business-focused production visibility linking scrap cost, likely failures, bottlenecks, delivery reliability, "
        "process parameters, material batches, and digital traceability."
    )

    with st.expander("Cost model used in this dashboard"):
        st.write(
            f"- Work cost: **€{LABOR_RATE_EUR_PER_HOUR:.0f}/hour**\n"
            f"- Setup cost: **€{SETUP_COST_PER_ORDER_EUR:.0f}/order**\n"
            f"- Material cost by kg: Aluminum €4, Stainless steel €2.8, Zinc/Galvanized steel €1.1, Basic steel €0.95\n"
            f"- Costs shown here are derived from the provided pricing logic and the production fields in the dataset."
        )

    # --------------------------------------------------
    # ALERTS
    # --------------------------------------------------
    if failed_units > 0:
        st.error(
            f"Scrap is currently affecting performance. {failed_units} units failed inspection, "
            f"with an estimated direct loss of €{direct_scrap_loss:,.0f}."
        )
    else:
        st.success("No failed units are currently visible in the selected view.")

    if len(high_risk_units) > 0:
        st.warning(
            f"{len(high_risk_units)} units are currently likely to fail. "
            f"Estimated business exposure: €{preventable_loss:,.0f}."
        )

    # --------------------------------------------------
    # KPI ROW
    # --------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Scrap Rate", f"{scrap_rate:.1f}%")
    k2.metric("Direct Scrap Loss", f"€{direct_scrap_loss:,.0f}")
    k3.metric("Predicted High-Risk Units", baseline_high_risk_count)
    k4.metric("Production Efficiency", f"{production_efficiency:.1f}%")
    k5.metric("Delivery Reliability", f"{delivery_reliability:.1f}%")

    st.divider()

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Executive Summary",
    "Units Likely to Fail",
    "Bottlenecks & Delivery",
    "Process Parameters & Batches",
    "Improvement Scenario",
    "Overall Solution",
    "Employee Reports"   # ✅ NEW TAB
])

    with tab1:
        c1, c2 = st.columns(2)

        top_scrap_process = (
            (filtered.groupby("process_step")["is_fail"].mean() * 100)
            .sort_values(ascending=False)
            .index[0]
            if len(filtered) > 0 else "N/A"
        )

        with c1:
            st.subheader("Scrap and efficiency")
            scrap_by_process = (filtered.groupby("process_step")["is_fail"].mean() * 100).sort_values(ascending=False)
            st.bar_chart(scrap_by_process)
            st.write(f"Scrap rate is **{scrap_rate:.1f}%**, and production efficiency is **{production_efficiency:.1f}%**.")
            st.success(f"Action: {solution_for_scrap(scrap_rate, top_scrap_process)}")

        with c2:
            st.subheader("Cost exposure")
            exposure_df = pd.DataFrame({
                "Category": ["Direct Scrap Loss", "High-Risk Exposure"],
                "EUR": [
                    float(direct_scrap_loss),
                    float(high_risk_units["estimated_business_impact_eur"].sum()) if not high_risk_units.empty else 0.0
                ]
            }).set_index("Category")
            st.bar_chart(exposure_df)
            st.success("Action: prioritize intervention on the highest-value units at risk first.")

    with tab2:
        st.subheader("Units likely to fail")
        if high_risk_units.empty:
            st.success("No high-risk units in the current filtered view.")
        else:
            cols = [
                "unit_id",
                "qr_code",
                "process_step",
                "machine_id",
                "batch_id",
                "risk_level",
                "predicted_failure_risk",
                "top_3_risk_drivers",
                "protective_drivers_to_preserve",
                "estimated_delay_min",
                "estimated_business_impact_eur",
                "suggested_solution"
            ]
            existing_cols = [c for c in cols if c in high_risk_units.columns]
            st.dataframe(high_risk_units[existing_cols], width="stretch")
            st.success("Action: start with the first rows in this table and apply the suggested solution before the next process step.")

    with tab3:
        left, right = st.columns(2)

        with left:
            st.subheader("Bottleneck machine")
            st.write(
                f"Current bottleneck: **{bottleneck_machine}** in **{slowing_process}**. "
                f"It is creating approximately **{delay_pct:.1f}%** more delay than the system average."
            )
            wait_by_machine = filtered.groupby("machine_id")["average_wait_time_min"].mean().sort_values(ascending=False)
            st.bar_chart(wait_by_machine)
            st.success(f"Action: {solution_for_bottleneck(bottleneck_machine, slowing_process)}")

        with right:
            st.subheader("Delivery reliability")
            if not quote_summary.empty:
                quote_display = quote_summary.copy()
                quote_display["total_hours"] = quote_display["total_hours"].round(2)
                quote_display["quoted_hours"] = quote_display["quoted_hours"].round(2)
                quote_display["on_time"] = quote_display["on_time"].map({True: "Yes", False: "No"})
                st.dataframe(quote_display[["quote_id", "total_hours", "quoted_hours", "on_time"]], width="stretch")
            else:
                st.info("No quote-linked records available in the current filtered view.")
            st.success("Action: combine bottleneck relief with early intervention on high-risk units to improve on-time delivery.")

    with tab4:
        p1, p2 = st.columns(2)

        with p1:
            st.subheader("Process parameter visibility")
            if not param_compare.empty:
                st.dataframe(param_compare.round(2), width="stretch")
            st.success("Action: reset unstable parameters toward the process median and monitor deviations automatically.")

        with p2:
            st.subheader("Material batch visibility")
            if not batch_summary.empty:
                st.dataframe(batch_summary.head(15).round(2), width="stretch")
            else:
                st.info("No batch information available in the current filtered view.")
            st.success("Action: flag high-risk batches for reinspection or batch-specific settings before the next stage.")

    with tab5:
        st.subheader("Machine improvement scenario")

        m1, m2 = st.columns(2)

        with m1:
            st.write(f"Selected machine: **{scenario_machine}**")
            st.write(f"Selected improvement: **{scenario_action}**")
            st.write("This scenario recalculates risk after improving a real operational condition on the selected machine.")

        with m2:
            st.metric("Current High-Risk Units", baseline_high_risk_count)
            st.metric("High-Risk Units After Improvement", scenario_high_risk_count)
            delta_units = baseline_high_risk_count - scenario_high_risk_count
            if delta_units > 0:
                st.success(f"Action: apply this improvement first. It reduces high-risk units by {delta_units}.")
            elif delta_units < 0:
                st.warning(f"This scenario worsens output by {abs(delta_units)} high-risk units. Avoid this change.")
            else:
                st.info("This scenario does not materially improve output. Prioritize another machine or another improvement type.")

    with tab6:
        st.subheader("Overall Business Impact & Solution")

        top_process = (
            (filtered.groupby("process_step")["is_fail"].mean() * 100)
            .sort_values(ascending=False)
            .index[0]
            if len(filtered) > 0 else "production"
        )

        top_machine = bottleneck_machine
        high_risk_pct = (baseline_high_risk_count / total_units * 100) if total_units > 0 else 0
        total_loss = float(direct_scrap_loss + preventable_loss + bottleneck_delay_cost)

        st.success(f"€{total_loss:,.0f} is currently at risk across scrap, delays, and high-risk units.")
        st.write(f"**{scrap_rate:.1f}%** scrap is primarily driven by **{top_process}**, while machine **{top_machine}** is contributing the highest delay.")
        st.write(f"**{high_risk_pct:.1f}%** of units are likely to fail before completion.")
        st.write(
            f"Stabilizing process conditions in **{top_process}** and improving performance of **{top_machine}** "
            f"can directly recover a significant portion of this cost and improve production efficiency beyond **{production_efficiency:.1f}%**.")
        # --------------------------------------------------
# EMPLOYEE REPORTS TAB (NEW)
# --------------------------------------------------
    with tab7:
        st.subheader("👷 Employee Welding Reports")

        reports = load_reports()

        if reports.empty:
            st.info("No reports submitted yet.")
        else:
            merged = reports.merge(
                data,
                on=["unit_id", "machine_id"],
                how="left"
            )

            merged["temp_deviation"] = abs(merged["actual_temperature"] - merged["ideal_temperature"])
            merged["pressure_deviation"] = abs(merged["actual_pressure"] - merged["ideal_pressure"])
            merged["power_deviation"] = abs(merged["actual_power"] - merged["ideal_power"])

            merged["severity_score"] = (
                merged["temp_deviation"] * 0.3 +
                merged["pressure_deviation"] * 0.3 +
                merged["power_deviation"] * 0.4
            )

            def impact(row):
                if row["severity_score"] > 50:
                    return "High scrap risk → potential rework or failure"
                elif row["severity_score"] > 20:
                    return "Moderate quality deviation → inspection needed"
                else:
                    return "Low impact"

            def solution(row):
                if row["issue_type"] == "burn_through":
                    return "Reduce temperature & welding speed immediately"
                if row["issue_type"] == "weak_joint":
                    return "Increase heat input or adjust pressure"
                if row["issue_type"] == "surface_porosity":
                    return "Check shielding gas and cleanliness"
                if row["issue_type"] == "machine_instability":
                    return "Inspect machine calibration and power supply"
                return "Monitor unit"

            merged["business_impact"] = merged.apply(impact, axis=1)
            merged["recommended_action"] = merged.apply(solution, axis=1)

            st.dataframe(
                merged[[
                    "unit_id", "machine_id", "issue_type",
                    "severity_score", "business_impact",
                    "recommended_action", "notes"
                ]],
                use_container_width=True
            )

    with st.expander("Show detailed risk drivers to reduce and protective drivers to preserve"):
        if not coef_df.empty:
            pos = coef_df.sort_values("coefficient", ascending=False).head(15).copy()
            neg = coef_df.sort_values("coefficient", ascending=True).head(15).copy()

            pos["business_meaning"] = pos["feature"].apply(readable_feature_name)
            neg["business_meaning"] = neg["feature"].apply(readable_feature_name)

            c1, c2 = st.columns(2)
            with c1:
                st.write("Risk drivers to reduce")
                st.dataframe(pos[["feature", "business_meaning", "coefficient"]], width="stretch")
            with c2:
                st.write("Protective drivers to preserve")
                st.dataframe(neg[["feature", "business_meaning", "coefficient"]], width="stretch")
        else:
            st.info("model_coefficients.csv not found.")
