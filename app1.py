# app1.py
import io
import os
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import requests

import streamlit as st

# Optional dependencies: matplotlib for plots, fpdf for PDF export
_have_matplotlib = True
_have_fpdf = True
try:
    import matplotlib.pyplot as plt
except Exception:
    _have_matplotlib = False
try:
    from fpdf import FPDF
except Exception:
    _have_fpdf = False

# --- Config
st.set_page_config(page_title="Machine Failure — Ensemble", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).parent
GITHUB_RAW_RELEASE_BASE = "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/models"

# -----------------------
# Helpers: download models from release if missing
# -----------------------

def download_asset_if_missing(rel_path: str, release_filename: str) -> Path | None:
    dest = ROOT / rel_path
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GITHUB_RAW_RELEASE_BASE}/{release_filename}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return dest
    except Exception:
        pass
    return None

# -----------------------
# Load models (cached)
# -----------------------
@st.cache_resource
def load_models() -> Dict[str, Any]:
    specs = {
        "Logistic Regression": ("models/final_model_pipeline_lr.pkl", "final_model_pipeline_lr.pkl"),
        "LightGBM": ("models/final_model_pipeline_lgb.pkl", "final_model_pipeline_lgb.pkl"),
        "Random Forest": ("models/final_model_pipeline_rfr.pkl", "final_model_pipeline_rfr.pkl"),
        "XGBoost": ("models/final_model_pipeline_xgb.pkl", "final_model_pipeline_xgb.pkl"),
    }
    loaded: Dict[str, Any] = {}
    for name, (rel_path, asset_name) in specs.items():
        p = ROOT / rel_path
        if not p.exists():
            download_asset_if_missing(rel_path, asset_name)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    loaded[name] = pickle.load(f)
            except Exception as e:
                loaded[name] = e
        else:
            loaded[name] = FileNotFoundError(f"Missing model file: {rel_path}")
    return loaded

models = load_models()

# -----------------------
# Safe predict helper
# -----------------------

def safe_predict(mdl, X: pd.DataFrame):
    """Return (preds_array_or_None, probs_array_or_None, error_or_None)"""
    try:
        preds = mdl.predict(X)
        preds = np.array(preds).ravel()
    except Exception as e:
        return None, None, e
    proba = None
    try:
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X)
            if getattr(p, "ndim", 0) == 2 and p.shape[1] >= 2:
                proba = np.array(p)[:, 1]
            else:
                proba = None
    except Exception:
        proba = None
    return preds, proba, None

# -----------------------
# UI
# -----------------------
st.title("⚙️ Machine Failure Prediction App")
st.markdown(
    "Upload sensor CSVs for batch predictions or use the sidebar for a single-sample prediction. App runs LR / LGB / RFR / XGB pipelines and shows an ensemble."
)

# Sidebar inputs
st.sidebar.header("Input Features (single sample)")
air_temp_c = st.sidebar.number_input("Air Temperature (°C)", value=300.0, step=1.0, format="%.2f")
process_temp_c = st.sidebar.number_input("Process Temperature (°C)", value=305.0, step=1.0, format="%.2f")
rot_speed = st.sidebar.number_input("Rotational Speed (rpm)", value=1800.0, step=1.0)
torque = st.sidebar.number_input("Torque (Nm)", value=40.0, step=1.0)
tool_wear = st.sidebar.number_input("Tool Wear (min)", value=150.0, step=1.0)

model_choice = st.sidebar.selectbox("Choose model for single-sample (or 'Ensemble')", options=["Ensemble"] + list(models.keys()))
st.sidebar.markdown("---")
allow_downloads = st.sidebar.checkbox("Allow per-file CSV download (batch)", value=True)
show_head = st.sidebar.checkbox("Show head preview (batch)", value=True)

# helper to build input df (pipelines expect Kelvin)
def build_input_df(air_c, proc_c, rpm, tq, wear):
    return pd.DataFrame({
        "Air temperature [K]": [air_c + 273.15],
        "Process temperature [K]": [proc_c + 273.15],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [tq],
        "Tool wear [min]": [wear],
    })

single_df = build_input_df(air_temp_c, process_temp_c, rot_speed, torque, tool_wear)

# Layout: two columns
col1, col2 = st.columns([1, 2])

# -----------------------
# Left: Single-sample
# -----------------------
with col1:
    st.header("Single-sample prediction")
    st.write("Use the sidebar inputs, choose `Ensemble` or a model and press Predict.")

    if st.button("Predict Failure"):
        st.info("Running predictions...")
        model_results: Dict[str, Dict[str, Any]] = {}
        for name, mdl in models.items():
            if isinstance(mdl, Exception):
                model_results[name] = {"error": str(mdl)}
                continue
            preds, proba, err = safe_predict(mdl, single_df)
            if err:
                model_results[name] = {"error": str(err)}
            else:
                model_results[name] = {
                    "pred": int(preds[0]) if preds is not None else None,
                    "proba": float(proba[0]) if proba is not None else None,
                }

        # compute ensemble
        probs = [v.get("proba") for v in model_results.values() if v.get("proba") is not None]
        if len(probs) > 0:
            ensemble_prob = float(np.nanmean(probs))
            ensemble_pred = int(ensemble_prob >= 0.5)
        else:
            preds_only = [v.get("pred") for v in model_results.values() if v.get("pred") is not None]
            if len(preds_only) > 0:
                ensemble_pred = int(round(np.mean(preds_only)))
                ensemble_prob = None
            else:
                ensemble_pred = None
                ensemble_prob = None

        # Build tidy table for display
        rows = []
        for name, res in model_results.items():
            if "error" in res:
                rows.append({"Model": name, "Prediction": "ERROR", "Prob": "—", "Notes": res.get("error")})
            else:
                pred = res.get("pred")
                prob = res.get("proba")
                rows.append({"Model": name, "Prediction": int(pred) if pred is not None else "—", "Prob": f"{prob:.3f}" if prob is not None else "—", "Notes": ""})

        df_results = pd.DataFrame(rows)
        st.markdown("### Model outputs")
        st.dataframe(df_results.set_index("Model"), use_container_width=True)

        st.markdown("### Ensemble result")
        if ensemble_pred is None:
            st.warning("No successful model outputs to produce an ensemble.")
        else:
            if ensemble_pred == 1:
                if ensemble_prob is not None:
                    st.error(f"🚨 Ensemble: MACHINE LIKELY TO FAIL (prob = {ensemble_prob:.2f})")
                else:
                    st.error("🚨 Ensemble: MACHINE LIKELY TO FAIL (by majority vote)")
            else:
                if ensemble_prob is not None:
                    st.success(f"✅ Ensemble: MACHINE SAFE (failure prob = {ensemble_prob:.2f})")
                else:
                    st.success("✅ Ensemble: MACHINE SAFE (by majority vote)")

        # probability bar chart (if matplotlib available and probabilities exist)
        if _have_matplotlib:
            model_names = []
            model_probs = []
            for name, res in model_results.items():
                if "error" in res:
                    continue
                p = res.get("proba")
                if p is not None:
                    model_names.append(name)
                    model_probs.append(p)
            if model_probs:
                fig, ax = plt.subplots(figsize=(5, 2.2))
                ax.barh(model_names, model_probs, height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Failure probability")
                ax.set_title("Model failure probabilities")
                for i, v in enumerate(model_probs):
                    ax.text(v + 0.01, i, f"{v:.2f}", va='center')
                st.pyplot(fig)
            else:
                st.info("No probability outputs available for bar chart (models may not expose predict_proba).")
        else:
            st.info("Plotting disabled: matplotlib not installed in the environment.")

# -----------------------
# Right: Batch section
# -----------------------
with col2:
    st.header("Batch prediction — CSV upload")
    st.write("Upload one or more CSV files with sensor features. The app will run all available models for each file and allow per-file download of results.")
    uploaded_files = st.file_uploader("Upload CSV files (accepts multiple)", type=["csv"], accept_multiple_files=True)

    all_results = []
    bad_files = []
    if uploaded_files:
        progress_bar = st.progress(0)
        total = len(uploaded_files)
        for idx, f in enumerate(uploaded_files, start=1):
            st.subheader(f.name)
            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                bad_files.append(f.name)
                continue

            if show_head:
                st.dataframe(df.head(5))

            preds_dict = {}
            probs_dict = {}
            for name, mdl in models.items():
                if isinstance(mdl, Exception):
                    preds_dict[name] = np.full(len(df), np.nan)
                    probs_dict[name] = np.full(len(df), np.nan)
                    continue
                preds, proba, err = safe_predict(mdl, df)
                if err:
                    preds_dict[name] = np.full(len(df), np.nan)
                    probs_dict[name] = np.full(len(df), np.nan)
                else:
                    preds_dict[name] = np.asarray(preds)
                    probs_dict[name] = np.asarray(proba) if proba is not None else np.full(len(df), np.nan)

            result_df = pd.DataFrame(index=np.arange(len(df)))
            for name in models.keys():
                result_df[name + "_pred"] = preds_dict.get(name, np.full(len(df), np.nan))
                result_df[name + "_prob"] = probs_dict.get(name, np.full(len(df), np.nan))

            prob_cols = [c for c in result_df.columns if c.endswith("_prob")]
            result_df["Ensemble_prob"] = result_df[prob_cols].mean(axis=1, skipna=True)
            if result_df["Ensemble_prob"].isna().all():
                pred_cols = [c for c in result_df.columns if c.endswith("_pred")]
                result_df["Ensemble_vote"] = result_df[pred_cols].mode(axis=1, numeric_only=True)[0]
            else:
                result_df["Ensemble_pred"] = (result_df["Ensemble_prob"] >= 0.5).astype(int)

            if "id" in df.columns:
                result_df.insert(0, "id", df["id"].values)
            result_df.insert(0, "source_file", f.name)

            all_results.append(result_df)

            if allow_downloads:
                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button(f"Download predictions for {f.name}", csv_bytes, file_name=f"{Path(f.name).stem}_predictions.csv")

            progress_bar.progress(idx / total)

        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            st.header("Combined predictions preview")
            st.dataframe(combined.head(30))
            st.download_button("Download combined CSV", combined.to_csv(index=False).encode(), "combined_predictions.csv")

# -----------------------
# PDF report
# -----------------------
st.markdown("---")
st.header("Export a PDF report (single-sample)")

if _have_fpdf:
    st.write("PDF export available.")
else:
    st.info("PDF export disabled: 'fpdf' not installed in this environment.")

if st.button("Create & download PDF report (single sample)"):
    model_results = {}
    for name, mdl in models.items():
        if isinstance(mdl, Exception):
            model_results[name] = {"error": str(mdl)}
        else:
            preds, proba, err = safe_predict(mdl, single_df)
            if err:
                model_results[name] = {"error": str(err)}
            else:
                model_results[name] = {"pred": int(preds[0]), "proba": float(proba[0]) if proba is not None else None}

    probs = [v.get("proba") for v in model_results.values() if v.get("proba") is not None]
    if len(probs) > 0:
        ensemble_prob = float(np.nanmean(probs))
        ensemble_pred = int(ensemble_prob >= 0.5)
    else:
        preds_only = [v.get("pred") for v in model_results.values() if v.get("pred") is not None]
        if len(preds_only) > 0:
            ensemble_pred = int(round(np.mean(preds_only)))
            ensemble_prob = None
        else:
            ensemble_pred = None
            ensemble_prob = None

    if not _have_fpdf:
        st.error("PDF export not available because fpdf is not installed.")
    else:
        # create pdf
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Machine Failure Prediction Report", ln=True, align='C')
        pdf.ln(6)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "Input (single sample):", ln=True)
        for k, v in single_df.iloc[0].items():
            pdf.cell(0, 7, f" - {k}: {v}", ln=True)
        pdf.ln(4)
        pdf.cell(0, 8, "Model outputs:", ln=True)
        for name, res in model_results.items():
            if "error" in res:
                pdf.cell(0, 6, f" - {name}: ERROR ({res['error']})", ln=True)
            else:
                if res.get("proba") is not None:
                    pdf.cell(0, 6, f" - {name}: pred={res['pred']}, prob={res['proba']:.3f}", ln=True)
                else:
                    pdf.cell(0, 6, f" - {name}: pred={res['pred']}", ln=True)
        pdf.ln(6)
        pdf.cell(0, 8, "Ensemble:", ln=True)
        if ensemble_pred is None:
            pdf.cell(0, 6, "No ensemble could be computed.", ln=True)
        else:
            pdf.cell(0, 6, f"Ensemble pred: {ensemble_pred}; prob: {ensemble_prob}", ln=True)

        out = io.BytesIO()
        out.write(pdf.output(dest='S').encode('latin1'))
        out.seek(0)
        st.download_button("Download PDF report", out, file_name="machine_failure_report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("If a model file is missing the app attempts to download it from the repo releases. For reliability, add the models/*.pkl into the `models/` folder in the repository and redeploy.")
