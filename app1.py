import streamlit as st
import numpy as np
import pickle

# --------------------------
# Load all 4 trained models
# --------------------------
with open("logistic_reg.pkl", "rb") as f:
    model_lr = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    model_rf = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model_xgb = pickle.load(f)

with open("lgbm_model.pkl", "rb") as f:
    model_lgbm = pickle.load(f)

# --------------------------
# App UI
# --------------------------
st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️")

st.title("Machine Failure Prediction App")
st.write("Enter the machine parameters below:")

# --------------------------
# User Inputs
# --------------------------
type_map = {"L": 0, "M": 1, "H": 2}

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature (°C)", value=300.0)
proc_temp = st.number_input("Process Temperature (°C)", value=310.0)
rot_speed = st.number_input("Rotational Speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (min)", value=100.0)

# Prepare input
input_data = np.array([
    type_map[machine_type],
    air_temp,
    proc_temp,
    rot_speed,
    torque,
    tool_wear
]).reshape(1, -1)

# --------------------------
# Prediction
# --------------------------
if st.button("Predict Failure"):
    pred_lr = model_lr.predict(input_data)[0]
    pred_rf = model_rf.predict(input_data)[0]
    pred_xgb = model_xgb.predict(input_data)[0]
    pred_lgbm = model_lgbm.predict(input_data)[0]

    st.subheader("Model Predictions")
    st.write(f"**Logistic Regression:** {'Failure' if pred_lr else 'No Failure'}")
    st.write(f"**Random Forest:** {'Failure' if pred_rf else 'No Failure'}")
    st.write(f"**XGBoost:** {'Failure' if pred_xgb else 'No Failure'}")
    st.write(f"**LightGBM:** {'Failure' if pred_lgbm else 'No Failure'}")

    # Majority vote
    total = pred_lr + pred_rf + pred_xgb + pred_lgbm
    final = "Machine Likely to Fail!" if total >= 2 else "✔️ Machine is Safe"

    st.subheader("Final Verdict")
    st.success(final) if total < 2 else st.error(final)
