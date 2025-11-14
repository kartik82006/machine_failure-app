# app1.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import requests
from pathlib import Path
import io
import sys
import time

st.set_page_config(page_title="Machine Failure Prediction App", layout="wide")
ROOT = Path(__file__).parent

st.title("⚙️ Machine Failure Prediction App")
st.markdown("Upload your machine parameters or use the sidebar inputs to get failure predictions from multiple models (LR / LGB / RFR / XGB).")

# -------------------------
# Model filenames (repo root)
# -------------------------
MODEL_FILENAMES = {
    "Logistic Regression": "final_model_pipeline_lr.pkl",
    "LightGBM": "final_model_pipeline_lgb.pkl",
    "Random Forest": "final_model_pipeline_rfr.pkl",
    "XGBoost": "final_model_pipeline_xgb.pkl",
}

# Attempt URL base for releases (if files are not in repo root)
# This assumes you uploaded assets to a release named "models" (your repo shows a release).
# If the release tag name is different, change the token below accordingly.
GITHUB_OWNER = "DURVA-GARGGG"
GITHUB_REPO = "machine_failure-app"
RELEASE_TAG = "models"  # your release shown in repo screenshots

def release_asset_url(filename: str) -> str:
    # e.g. https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/models/final_model_pipeline_xgb.pkl
    return f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{filename}"

def download_to_path(url: str, out_path: Path, timeout=30):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception as e:
        return False

@st.cache_resource
def load_models():
    """
    Returns dict: {name: model_or_exception}
    Loads from local repo root; if not found, attempts to download from release URL.
    """
    models = {}
    for name, fname in MODEL_FILENAMES.items():
        p = ROOT / fname
        # If not present locally, try to download from release asset
        if not p.exists():
            url = release_asset_url(fname)
            try:
                ok = download_to_path(url, p)
                if not ok:
                    raise FileNotFoundError(f"Not found locally and failed to download {fname}")
            except Exception as e:
                models[name] = e
                continue
        # Now attempt to load
        try:
            # joblib.load handles sklearn pipelines well
            m = joblib.load(p)
            models[name] = m
        except Exception as e1:
            # try pickle as fallback
            try:
                with open(p, "rb") as f:
                    m = pickle.load(f)
                models[name] = m
            except Exception as e2:
                models[name] = Exception(f"Failed to load model file {p.name}: {e2}")
    return models

models = load_models()

# Sidebar: single-sample input
st.sidebar.header("Input Features (single sample)")
air_temp_c = st.sidebar.number_input("Air Temperature (°C)", min_value=0.0, max_value=1000.0, value=300.0, step=1.0)
process_temp_c = st.sidebar.number_input("Process Temperature (°C)", min_value=0.0, max_value=1000.0, value=305.0, step=1.0)
rot_speed = st.sidebar.number_input("Rotational Speed (rpm)", min_value=0.0, max_value=10000.0, value=1800.0, step=10.0)
torque = st.sidebar.number_input("Torque (Nm)", min_value=0.0, max_value=10000.0, value=40.0, step=1.0)
tool_wear = st.sidebar.number_input("Tool Wear (min)", min_value=0.0, max_value=10000.0, value=150.0, step=1.0)

model_choice = st.sidebar.selectbox("Choose Prediction Model (single sample)", options=list(models.keys()))

st.sidebar.markdown("**Model status:**")
for nm, mv in models.items():
    if isinstance(mv, Exception):
        st.sidebar.error(f"{nm}: ❌")
    else:
        st.sidebar.success(f"{nm}: ✅")

# Helper: prepare input DataFrame matching the pipelines used
def make_input_df(air_c, proc_c, rot, tq, tw):
    # Many datasets use Kelvin for temps; your old code added 273 — keep that behavior
    return pd.DataFrame({
        "Air temperature [K]": [air_c + 273.0],
        "Process temperature [K]": [proc_c + 273.0],
        "Rotational speed [rpm]": [rot],
        "Torque [Nm]": [tq],
        "Tool wear [min]": [tw],
    })

# Prediction helpers (handle models missing predict_proba)
def predict_with_model(model, X):
    try:
        preds = model.predict(X)
    except Exception as e:
        raise Exception(f"Error in predict(): {e}")
    prob = None
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            # if binary, take second column
            if prob.shape[1] >= 2:
                prob = prob[:,1]
            else:
                prob = prob.ravel()
        else:
            # fallback: decision_function -> map to pseudo-prob via sigmoid
            if hasattr(model, "decision_function"):
                df = model.decision_function(X)
                # sigmoid
                prob = 1/(1+np.exp(-df))
            else:
                prob = None
    except Exception:
        prob = None
    return np.asarray(preds).ravel(), (np.asarray(prob).ravel() if prob is not None else None)

# Main layout: two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Single-sample prediction")
    st.write("Use the sidebar inputs, choose a model, then press Predict.")
    if st.button("Predict Failure"):
        chosen_model = models.get(model_choice)
        if isinstance(chosen_model, Exception):
            st.error(f"Chosen model not available: {chosen_model}")
        else:
            X = make_input_df(air_temp_c, process_temp_c, rot_speed, torque, tool_wear)
            try:
                preds, probs = predict_with_model(chosen_model, X)
                pred = int(preds[0])
                prob_str = f"{probs[0]:.3f}" if (probs is not None and len(probs)>0 and not np.isnan(probs[0])) else "N/A"
                if pred == 1:
                    st.error(f"Prediction: **FAILURE**  —  Probability ≈ {prob_str}")
                else:
                    st.success(f"Prediction: **SAFE**  —  Failure probability ≈ {prob_str}")
                st.write("---")
                st.write("**Input data used:**")
                st.dataframe(X)
            except Exception as e:
                st.exception(f"Error while predicting: {e}")

with col2:
    st.header("Batch prediction — CSV upload")
    st.markdown("Upload one or more CSV files containing the sensor features. The app will run all available models and produce per-file CSV downloads.")
    uploaded_files = st.file_uploader("Upload CSV files (accepts multiple)", type=["csv"], accept_multiple_files=True)
    allow_downloads = st.checkbox("Allow per-file download", value=True)
    show_preview = st.checkbox("Show head preview (5 rows)", value=True)

    if uploaded_files:
        all_results = []
        for f in uploaded_files:
            st.subheader(f"File: {f.name}")
            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                continue

            if show_preview:
                st.dataframe(df.head(5))

            # run each model
            model_preds = {}
            numeric_preds_for_ensemble = {}
            errors_present = False
            for name, mdl in models.items():
                if isinstance(mdl, Exception):
                    model_preds[name] = f"Error: {mdl}"
                    errors_present = True
                else:
                    try:
                        preds = mdl.predict(df)
                        preds = np.asarray(preds).ravel()
                        model_preds[name] = preds
                        numeric_preds_for_ensemble[name] = preds
                    except Exception as e:
                        model_preds[name] = f"Error during prediction: {e}"
                        errors_present = True

            if len(numeric_preds_for_ensemble) == 0:
                st.error("No successful numeric predictions for this file. See model messages below.")
                st.json({k: str(v) for k,v in model_preds.items()})
                continue

            # Construct results DataFrame (one row per sample)
            n = len(next(iter(numeric_preds_for_ensemble.values())))
            result_df = pd.DataFrame(index=np.arange(n))
            for name in MODEL_FILENAMES.keys():
                val = model_preds.get(name)
                if isinstance(val, (str, Exception)):
                    result_df[name] = np.nan
                else:
                    result_df[name] = np.asarray(val)

            # ensemble average across available numeric cols
            result_df["Ensemble_Average"] = result_df.mean(axis=1)

            # attach id column if present
            idcol = None
            for c in ["id","Id","ID","machine_id","machineID"]:
                if c in df.columns:
                    idcol = c
                    break
            if idcol:
                result_df.insert(0, idcol, df[idcol].values)

            st.success("Predictions ready ✅")
            st.dataframe(result_df.head(6))

            csv_bytes = result_df.to_csv(index=False).encode()
            if allow_downloads:
                st.download_button(f"Download predictions for {f.name}", csv_bytes, file_name=f"{Path(f.name).stem}_predictions.csv")

            # keep for combined
            result_df.insert(0, "source_file", f.name)
            all_results.append(result_df)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            st.header("Combined predictions (all uploaded files)")
            st.dataframe(combined.head(10))
            st.download_button("Download combined CSV", combined.to_csv(index=False).encode(), file_name="combined_predictions.csv")

st.markdown("---")
st.caption("If models failed to load, open Manage app → Logs in Streamlit Cloud to see detailed errors. If a model file was missing, the app attempted to download it from the Release assets.")

# Small local run helper (for debugging)
if st.sidebar.button("Show internal model load status (debug)"):
    debug_status = {k: (str(type(v)) if not isinstance(v, Exception) else f"ERROR: {v}") for k,v in models.items()}
    st.sidebar.json(debug_status)
