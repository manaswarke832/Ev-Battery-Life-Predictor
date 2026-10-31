
import streamlit as st
import pandas as pd
import joblib

# Load the productionized automated inference pipeline object file
try:
    production_pipeline = joblib.load('optimized_ev_battery_pipeline.pkl')
except Exception as loading_error:
    st.error(f"Failed to extract the integrated pipeline binary component: {loading_error}")

# Set page configurations
st.set_page_config(page_title="EV Battery Diagnostic Panel", layout="centered")

st.title(" Integrated EV Battery Degradation Predictor Dashboard")
st.markdown("This dashboard implements an end-to-end operational Machine Learning pipeline running an optimized regression framework to assess battery capacity drop over active automotive telematics parameters.")

st.markdown("---")
st.subheader(" Step-1: Input Operational Fleet Telemetry Records")

# Splitting inputs into structured parallel grid columns
left_column, right_column = st.columns(2)

with left_column:
    car_model_input = st.selectbox("Vehicle Architecture Variant", ["Tesla Mod NMC", "Ford Must NMC", "Hyundai Ic NMC", "Wuling Air LFP", "BYD Atto LFP"])
    battery_type_input = st.selectbox("Battery Chemistry Profile", ["NMC", "LFP"])
    driving_style_input = st.selectbox("Driver Behavior Logging Classification", ["Aggressive", "Conservative", "Moderate"])
    vehicle_age_input = st.slider("Active Operation Lifespan Duration (Months)", 0, 120, 36)

with right_column:
    charging_cycles_input = st.number_input("Total Lifecycle Charging Events Counter", min_value=0, max_value=3000, value=400)
    ambient_temp_input = st.number_input("Average Thermal Ambient Operating Temp (°C)", min_value=-20.0, max_value=60.0, value=28.5)
    fast_charging_ratio = st.slider("High-Voltage Fast Charging Utilization Share", 0.0, 1.0, 0.35)
    discharge_rate_input = st.number_input("Average Load Discharge Velocity Speed", min_value=0.0, max_value=15.0, value=1.65)

resistance_input = st.number_input("Cell Pack Internal Resistance Rating (Ω Metric)", min_value=0.0000, max_value=0.2000, value=0.0340, format="%.4f")

st.markdown("---")

# Execution button trigger
if st.button("Evaluate Battery State of Health", type="primary"):
    # Structure input features to exactly replicate the core dataframe scheme headers
    raw_telemetry_payload = pd.DataFrame([{
        'Car_Mode': car_model_input,
        'Battery_Ty': battery_type_input,
        'Vehicle_Ag': vehicle_age_input,
        'Total_Cha': charging_cycles_input,
        'Avg_Temp': ambient_temp_input,
        'Fast_Char': fast_charging_ratio,
        'Avg_Disch': discharge_rate_input,
        'Driving_St': driving_style_input,
        'Internal_R': resistance_input
    }])

    with st.spinner("Executing dynamic column matrix preprocessing conversions..."):
        try:
            # Generate predictions using the runtime instance
            predicted_soh_value = production_pipeline.predict(raw_telemetry_payload)[0]

            # Render numerical validation results
            st.metric(label="Calculated Remaining Battery State of Health (SoH %)", value=f"{predicted_soh_value:.2f}%")

            # Visual alerts categorization
            if predicted_soh_value >= 85.0:
                st.success(" **System Health Diagnosis**: Optimal Condition Status (Low Degradation Class)")
            elif predicted_soh_value >= 70.0:
                st.warning(" **System Health Diagnosis**: Moderate Capacity Deficit Discovered (Standard Life Cycle Aging)")
            else:
                st.error(" **System Health Diagnosis**: Critical Performance Capacity Alert! Cell Integrity Inspection Highly Recommended.")

        except Exception as runtime_error:
            st.error(f"Execution Error Encountered During Inference Array Transform: {runtime_error}")
