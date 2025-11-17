# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import requests

st.set_page_config(page_title="Machine Failure — Ensemble Predictions", layout="wide")

###########################################
# DOWNLOAD MODELS FROM RELEASES
###########################################

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_URLS = {
    "LightGBM": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_lgb.pkl",
    "Logistic Regression": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_lr.pkl",
    "Random Forest": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_rfr.pkl",
    "XGBoost": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_xgb.pkl",
}

def download_models():
    for name, url in MODEL_URLS.items():
        filename = MODEL_DIR / url.split("/")[-1]
        if not filename.exists():
            r = requests.get(url)
            open(filename, "wb").write(r.content)

download_models()

###########################################
# LOAD MODELS
###########################################

@st.cache_resource
def load_models():
    models = {}
    for name, url in MODEL_URLS.items():
        filename = MODEL_DIR / url.split("/")[-1]
        try:
            models[name] = joblib.load(filename)
        except Exception as e:
            models[name] = e
    return models

models = load_models()

###########################################
# STREAMLIT UI
###########################################

st.title("Machine Failure Prediction — Ensemble Models")
st.write("Upload CSV files to generate predictions.")

uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload CSV files to continue.")
    st.stop()

show_head = st.checkbox("Show first 5 rows", value=True)

all_results = []

for f in uploaded_files:
    st.subheader(f.name)

    try:
        df = pd.read_csv(f)
    except Exception as e:
        st.error(f"Error reading {f.name}: {e}")
        continue

    if show_head:
        st.dataframe(df.head())

    preds_dict = {}

    for name, mdl in models.items():
        try:
            preds = mdl.predict(df)
            preds_dict[name] = np.array(preds).ravel()
        except Exception as e:
            preds_dict[name] = f"Error: {e}"

    # Build result DataFrame
    result_df = pd.DataFrame(preds_dict)

    # Ensemble
    numeric_cols = result_df.select_dtypes(include=[np.number])
    result_df["Ensemble_Average"] = numeric_cols.mean(axis=1)

    st.dataframe(result_df.head())

    csv = result_df.to_csv(index=False).encode()
    st.download_button(f"Download results for {f.name}", csv, f"{f.name}_predictions.csv")

    result_df.insert(0, "source_file", f.name)
    all_results.append(result_df)

if all_results:
    final_output = pd.concat(all_results, ignore_index=True)
    st.header("Combined Predictions")
    st.dataframe(final_output.head())

    st.download_button(
        "Download Combined CSV",
        final_output.to_csv(index=False).encode(),
        "combined_predictions.csv",
    )
