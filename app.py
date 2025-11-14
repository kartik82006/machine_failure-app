# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import requests

def download_file(url, filename):
    r = requests.get(url)
    open(filename, 'wb').write(r.content)
download_file(
    "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_xgb.pkl",
    "final_model_pipeline_xgb.pkl"
)


st.set_page_config(page_title="Machine Failure — Ensemble Predictions", layout="wide")

st.title("Machine Failure Prediction — Ensemble (LGB / LR / RFR / XGB)")
st.markdown("Upload one or more CSV sensor files. The app runs 4 saved pipelines and shows model-wise predictions + an ensemble average.")

ROOT = Path(__file__).parent

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "LightGBM": "models/final_model_pipeline_lgb.pkl",
        "Logistic Regression": "models/final_model_pipeline_lr.pkl",
        "Random Forest": "models/final_model_pipeline_rfr.pkl",
        "XGBoost": "models/final_model_pipeline_xgb.pkl",
    }
    for name, rel in model_files.items():
        p = ROOT / rel
        if p.exists():
            try:
                models[name] = joblib.load(p)
            except Exception as e:
                models[name] = e
        else:
            models[name] = FileNotFoundError(f"Missing file: {p}")
    return models

models = load_models()

st.sidebar.header("Options")
show_head = st.sidebar.checkbox("Show file preview head (5 rows)", value=True)
download_individual = st.sidebar.checkbox("Allow individual file download", value=True)

uploaded_files = st.file_uploader("Upload CSV files (sensor data)", type=["csv"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload CSV files to run predictions.")
    st.stop()

all_results = []
bad_files = []

for f in uploaded_files:
    st.subheader(f" {f.name}")
    try:
        df = pd.read_csv(f)
    except Exception as e:
        st.error(f"Could not read {f.name}: {e}")
        bad_files.append(f.name)
        continue

    if show_head:
        st.dataframe(df.head(5))

    preds_dict = {}
    # run each model
    for name, mdl in models.items():
        if isinstance(mdl, Exception):
            preds_dict[name] = f"Error: {mdl}"
        else:
            try:
                # Many sklearn pipelines expect a DataFrame or numpy
                preds = mdl.predict(df)
                # If preds is single value, broadcast
                preds = np.array(preds).ravel()
                preds_dict[name] = preds
            except Exception as e:
                preds_dict[name] = f"Error during prediction: {e}"

    # Build result dataframe across rows
    # If any model returned an error string, show message instead of numeric.
    numeric_preds = {}
    errors_present = False
    for name, val in preds_dict.items():
        if isinstance(val, (str, Exception)):
            errors_present = True
        else:
            numeric_preds[name] = np.asarray(val)

    if errors_present and len(numeric_preds) == 0:
        st.error("All models failed for this file. See model messages below.")
        st.json({k: str(v) for k, v in preds_dict.items()})
        continue

    # Build dataframe where each row corresponds to a sample
    # Use numeric_preds to compute ensemble; for model columns that errored, fill with NaN
    n_rows = None
    for v in numeric_preds.values():
        n_rows = len(v)
        break
    if n_rows is None:
        st.error("Could not determine number of rows from predictions.")
        continue

    result_df = pd.DataFrame(index=np.arange(n_rows))
    for name in models.keys():
        val = preds_dict.get(name)
        if isinstance(val, (str, Exception)):
            result_df[name] = np.nan
        else:
            result_df[name] = np.asarray(val)

    # Ensemble strategy: mean of available numeric predictions (skip NaNs)
    result_df["Ensemble_Average"] = result_df.mean(axis=1)

    # Attach identifier from original df if exists (e.g., id column)
    # If there's a column named 'id' or 'Id' or 'ID', attach it
    id_col = None
    for candidate in ["id", "Id", "ID", "machine_id", "machineID"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col:
        result_df.insert(0, id_col, df[id_col].values)

    st.success("Predictions generated ✔️")
    st.dataframe(result_df.head(10) if show_head else result_df)

    # allow download of per-file results
    csv_bytes = result_df.to_csv(index=False).encode()
    if download_individual:
        st.download_button(f"Download predictions for {f.name}", csv_bytes, file_name=f"{Path(f.name).stem}_predictions.csv")

    # append to final
    # add source filename column
    result_df.insert(0, "source_file", f.name)
    all_results.append(result_df)

# Combine all results
if all_results:
    final_output = pd.concat(all_results, ignore_index=True)
    st.header("Combined Predictions for all files")
    st.dataframe(final_output.head(20))
    st.download_button("Download combined predictions CSV", final_output.to_csv(index=False).encode(), "combined_predictions.csv")
else:
    st.warning("No successful predictions to combine.")

st.markdown("---")
st.caption("If a model raised errors, check model files in the `models/` folder. Pipelines provided in this repo should contain preprocessing + estimator.")


