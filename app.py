import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Stremet Risk & Tracking Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD FILES
# -----------------------------
data = pd.read_csv("production_data.csv")
model = joblib.load("defect_model.pkl")
model_columns = joblib.load("model_columns.pkl")

if os.path.exists("model_coefficients.csv"):
    coef_df = pd.read_csv("model_coefficients.csv")
else:
    coef_df = pd.DataFrame({"feature": [], "coefficient": []})

# -----------------------------
# BASIC PREP
# -----------------------------
data["is_fail"] = (data["result"] == "fail").astype(int)

# Safety: ensure needed columns exist
required_cols = [
    "unit_id", "qr_code", "process_step", "machine_id",
    "time_taken_min", "queue_length_units", "average_wait_time_min",
    "temperature_C", "pressure_bar", "power_usage_kW",
    "material_type", "thickness_mm", "estimated_time_hours",
    "estimated_cost_eur", "risk_score_raw", "formula_risk_probability"
]
for col in required_cols:
    if col not in data.columns:
        data[col] = 0

# -----------------------------
# HEADER
# -----------------------------
st.title("Stremet Risk & Tracking Dashboard")
st.write("QR-based traceability, role-based access, real-time risk monitoring, and what-if simulation.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Access")

user_role = st.sidebar.selectbox(
    "Select User View",
    ["Operator", "Quality Engineer", "Manager"]
)

selected_qr = st.sidebar.selectbox(
    "Scan / Select QR Code",
    sorted(data["qr_code"].dropna().unique())
)

selected_unit_df = data[data["qr_code"] == selected_qr].copy()
selected_unit = selected_unit_df["unit_id"].iloc[0] if not selected_unit_df.empty else "N/A"

st.sidebar.header("Filters")

selected_steps = st.sidebar.multiselect(
    "Process Step",
    sorted(data["process_step"].dropna().unique()),
    default=sorted(data["process_step"].dropna().unique())
)

selected_machines = st.sidebar.multiselect(
    "Machine ID",
    sorted(data["machine_id"].dropna().unique()),
    default=sorted(data["machine_id"].dropna().unique())
)

selected_materials = st.sidebar.multiselect(
    "Material Type",
    sorted(data["material_type"].dropna().unique()),
    default=sorted(data["material_type"].dropna().unique())
)

filtered = data[
    data["process_step"].isin(selected_steps) &
    data["machine_id"].isin(selected_machines) &
    data["material_type"].isin(selected_materials)
].copy()

# -----------------------------
# GLOBAL ALERTS
# -----------------------------
high_risk_units = filtered[filtered["formula_risk_probability"] >= 0.60]
medium_risk_units = filtered[
    (filtered["formula_risk_probability"] >= 0.40) &
    (filtered["formula_risk_probability"] < 0.60)
]

if len(high_risk_units) > 0:
    st.error(
        f"🚨 {len(high_risk_units)} high-risk production events detected. Immediate inspection recommended."
    )
elif len(medium_risk_units) > 0:
    st.warning(
        f"⚠️ {len(medium_risk_units)} medium-risk events detected. Monitor closely."
    )
else:
    st.success("✅ All visible units are currently within low-risk range.")

# -----------------------------
# KPI ROW
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Events", len(filtered))
c2.metric("Unique Units", filtered["unit_id"].nunique())
c3.metric("Observed Fail Rate", f"{100 * filtered['is_fail'].mean():.1f}%")
c4.metric("Avg Formula Risk", f"{100 * filtered['formula_risk_probability'].mean():.1f}%")
c5.metric("Avg Queue Length", f"{filtered['queue_length_units'].mean():.1f}")

st.divider()

# -----------------------------
# TOP DASHBOARD SUMMARY
# -----------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("🔥 Top 5 Risk Units")
    top_units = filtered.sort_values("formula_risk_probability", ascending=False).head(5)
    st.dataframe(
        top_units[[
            "unit_id", "qr_code", "process_step", "machine_id",
            "material_type", "queue_length_units",
            "time_taken_min", "formula_risk_probability"
        ]],
        width="stretch"
    )

with right_col:
    st.subheader("💡 Key Insights")
    risk_by_process_summary = (
        filtered.groupby("process_step")["formula_risk_probability"]
        .mean()
        .sort_values(ascending=False)
    )

    if not risk_by_process_summary.empty:
        worst_step = risk_by_process_summary.idxmax()
        worst_risk = risk_by_process_summary.max()
        st.info(
            f"Highest average risk is in **{worst_step}** "
            f"at **{worst_risk:.1%}**."
        )

    queue_by_process_summary = (
        filtered.groupby("process_step")["queue_length_units"]
        .mean()
        .sort_values(ascending=False)
    )
    if not queue_by_process_summary.empty:
        queue_hotspot = queue_by_process_summary.idxmax()
        st.write(f"**Queue hotspot:** {queue_hotspot}")

    fail_by_machine_summary = (
        filtered.groupby("machine_id")["is_fail"]
        .mean()
        .sort_values(ascending=False)
    )
    if not fail_by_machine_summary.empty:
        worst_machine = fail_by_machine_summary.idxmax()
        st.write(f"**Highest failure-rate machine:** {worst_machine}")

st.divider()

# -----------------------------
# LIVE VISUALS
# -----------------------------
v1, v2, v3 = st.columns(3)

with v1:
    st.subheader("📍 Unit Flow Status")
    flow_counts = filtered.groupby("process_step")["unit_id"].count().sort_values(ascending=False)
    st.bar_chart(flow_counts)

with v2:
    st.subheader("⚠️ Risk by Process Stage")
    risk_by_process = (
        filtered.groupby("process_step")["formula_risk_probability"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(risk_by_process)

with v3:
    st.subheader("⏱️ Average Queue by Process")
    queue_by_process = (
        filtered.groupby("process_step")["queue_length_units"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(queue_by_process)

st.divider()

# -----------------------------
# ROLE-BASED VIEWS
# -----------------------------
if user_role == "Operator":
    st.subheader("Operator View")
    st.write(f"Selected Unit: **{selected_unit}**")

    op_col1, op_col2 = st.columns([1.2, 1])

    with op_col1:
        operator_cols = [
            "unit_id", "qr_code", "process_step", "machine_id",
            "start_time", "end_time", "status",
            "storage_location", "recommended_device"
        ]
        operator_cols = [c for c in operator_cols if c in selected_unit_df.columns]
        st.dataframe(selected_unit_df[operator_cols], width="stretch")

    with op_col2:
        if not selected_unit_df.empty:
            latest = selected_unit_df.iloc[-1]
            current_step = latest["process_step"]
            next_step_map = {
                "laser_cutting": "bending",
                "bending": "welding",
                "welding": "inspection / packaging"
            }
            next_step = next_step_map.get(current_step, "completed")

            st.metric("Current Stage", current_step)
            st.metric("Current Risk", f"{latest['formula_risk_probability']:.1%}")
            st.info(f"Next suggested step: **{next_step}**")

            if latest["formula_risk_probability"] > 0.5:
                st.warning("Please notify quality team before the next stage.")
            else:
                st.success("Unit can proceed normally.")

elif user_role == "Quality Engineer":
    tab1, tab2, tab3 = st.tabs([
        "Inspection Priority",
        "Unit Traceability",
        "What-if Simulation"
    ])

    with tab1:
        st.subheader("Inspection Priority List")
        priority = filtered.sort_values("formula_risk_probability", ascending=False).head(20)
        cols = [
            "unit_id", "qr_code", "process_step", "machine_id",
            "material_type", "time_taken_min", "queue_length_units",
            "result", "defect_type", "severity", "formula_risk_probability"
        ]
        cols = [c for c in cols if c in priority.columns]
        st.dataframe(priority[cols], width="stretch")

    with tab2:
        st.subheader("Unit Traceability")
        cols = [
            "unit_id", "qr_code", "process_step", "machine_id",
            "time_taken_min", "queue_length_units", "average_wait_time_min",
            "temperature_C", "pressure_bar", "power_usage_kW",
            "result", "defect_type", "severity", "formula_risk_probability"
        ]
        cols = [c for c in cols if c in selected_unit_df.columns]
        st.dataframe(selected_unit_df[cols], width="stretch")

    with tab3:
        st.subheader("What-if Simulation")

        if not selected_unit_df.empty:
            base_row = selected_unit_df.iloc[-1]

            sim1, sim2, sim3 = st.columns(3)

            with sim1:
                sim_machine = st.selectbox(
                    "Simulate Machine",
                    sorted(data[data["process_step"] == base_row["process_step"]]["machine_id"].unique())
                )

            with sim2:
                sim_queue = st.slider(
                    "Simulate Queue Length",
                    0, 20, int(base_row["queue_length_units"])
                )

            with sim3:
                sim_time_factor = st.slider(
                    "Simulate Processing Time Factor",
                    0.7, 1.5, 1.0, 0.05
                )

            ref = data[
                (data["process_step"] == base_row["process_step"]) &
                (data["machine_id"] == sim_machine)
            ]

            if ref.empty:
                ref = data[data["process_step"] == base_row["process_step"]]

            process_step = base_row["process_step"]
            material_type = base_row["material_type"]
            thickness_mm = float(base_row["thickness_mm"])
            estimated_time_hours = float(base_row["estimated_time_hours"])
            estimated_cost_eur = float(base_row["estimated_cost_eur"])

            time_taken_min = ref["time_taken_min"].median() * sim_time_factor
            average_wait_time_min = ref["average_wait_time_min"].median()
            temperature_C = ref["temperature_C"].median()
            pressure_bar = ref["pressure_bar"].median()
            power_usage_kW = ref["power_usage_kW"].median()

            process_complexity_map = {
                "laser_cutting": 0.55,
                "bending": 0.70,
                "welding": 1.00
            }

            material_difficulty_map = {
                "stainless_steel_304": 1.00,
                "galvanized_steel": 0.75,
                "aluminum_5754": 0.55
            }

            ref_time = data.groupby("process_step")["time_taken_min"].median().to_dict()
            ref_queue = data.groupby("process_step")["queue_length_units"].median().to_dict()
            ref_wait = data.groupby("process_step")["average_wait_time_min"].median().to_dict()
            ref_temp = data.groupby("process_step")["temperature_C"].median().to_dict()
            ref_power = data.groupby("process_step")["power_usage_kW"].median().to_dict()

            time_ratio = time_taken_min / ref_time.get(process_step, 10)
            queue_ratio = sim_queue / ref_queue.get(process_step, 5)
            wait_ratio = average_wait_time_min / ref_wait.get(process_step, 15)
            temp_ratio = temperature_C / ref_temp.get(process_step, 100)
            power_ratio = power_usage_kW / ref_power.get(process_step, 8)

            thickness_min = data["thickness_mm"].min()
            thickness_max = data["thickness_mm"].max()
            thickness_norm = 0 if thickness_max == thickness_min else (
                (thickness_mm - thickness_min) / (thickness_max - thickness_min)
            )

            operational_stress = time_ratio * queue_ratio * temp_ratio * power_ratio
            risk_score_raw = (
                operational_stress *
                (1 + 0.5 * thickness_norm) *
                material_difficulty_map[material_type] *
                process_complexity_map[process_step]
            )

            risk_median = data["risk_score_raw"].median()
            formula_risk_probability = 1 / (1 + pow(2.718281828, -((risk_score_raw - risk_median) * 2)))

            input_df = pd.DataFrame([{
                "process_step": process_step,
                "machine_id": sim_machine,
                "time_taken_min": time_taken_min,
                "queue_length_units": sim_queue,
                "average_wait_time_min": average_wait_time_min,
                "temperature_C": temperature_C,
                "pressure_bar": pressure_bar,
                "power_usage_kW": power_usage_kW,
                "material_type": material_type,
                "thickness_mm": thickness_mm,
                "estimated_time_hours": estimated_time_hours,
                "estimated_cost_eur": estimated_cost_eur,
                "risk_score_raw": risk_score_raw
            }])

            input_encoded = pd.get_dummies(
                input_df,
                columns=["process_step", "machine_id", "material_type"],
                drop_first=False
            )

            for col in model_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            input_encoded = input_encoded[model_columns]
            pred_prob = model.predict_proba(input_encoded)[0][1]

            s1, s2, s3 = st.columns(3)

            with s1:
                st.metric(
                    "Original Risk",
                    f"{base_row['formula_risk_probability']:.1%}"
                )

            with s2:
                st.metric(
                    "Simulated Formula Risk",
                    f"{formula_risk_probability:.1%}"
                )

            with s3:
                st.metric(
                    "Simulated Model Defect Risk",
                    f"{pred_prob:.1%}"
                )

            st.subheader("Simulation Impact")
            if pred_prob > base_row["formula_risk_probability"]:
                st.warning("Risk increased under this scenario.")
            else:
                st.success("Risk decreased or stayed stable under this scenario.")

            if pred_prob > 0.5:
                st.error("Suggested action: inspect this unit before the next step.")
            else:
                st.success("Suggested action: unit can proceed with routine monitoring.")

else:  # Manager
    tab1, tab2, tab3 = st.tabs([
        "KPIs",
        "Bottlenecks",
        "Model Drivers"
    ])

    with tab1:
        st.subheader("Production KPIs")
        m1, m2 = st.columns(2)

        with m1:
            st.write("Average Risk by Process")
            chart = (
                filtered.groupby("process_step")["formula_risk_probability"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(chart)

        with m2:
            st.write("Failure Rate by Machine")
            fail_rate = (
                filtered.groupby("machine_id")["is_fail"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(fail_rate)

    with tab2:
        st.subheader("Bottleneck Overview")
        b1, b2 = st.columns(2)

        with b1:
            st.write("Average Queue Length by Process")
            queue_chart = (
                filtered.groupby("process_step")["queue_length_units"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(queue_chart)

        with b2:
            st.write("Average Processing Time by Process")
            time_chart = (
                filtered.groupby("process_step")["time_taken_min"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(time_chart)

        bottleneck_step = filtered.groupby("process_step")["time_taken_min"].mean().idxmax()
        bottleneck_value = filtered.groupby("process_step")["time_taken_min"].mean().max()
        st.info(
            f"Current bottleneck stage: **{bottleneck_step}** "
            f"with **{bottleneck_value:.2f} min** average processing time."
        )

    with tab3:
        st.subheader("Model Drivers")
        if not coef_df.empty:
            pos, neg = st.columns(2)
            with pos:
                st.write("Top Positive Risk Drivers")
                st.dataframe(coef_df.head(15), width="stretch")
            with neg:
                st.write("Top Negative Risk Drivers")
                st.dataframe(
                    coef_df.tail(15).sort_values("coefficient", ascending=True),
                    width="stretch"
                )
        else:
            st.info("model_coefficients.csv not found.")
