# app.py
import io
import os
import sys
import time
import math
import pickle
import requests
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from fpdf import FPDF

st.set_page_config(page_title="Machine Failure — Ensemble", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).parent

# -----------------------
# Utility: download model from GitHub release if not present
# -----------------------
GITHUB_RAW_RELEASE_BASE = "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/models"

def download_asset_if_missing(rel_path: str, release_filename: str):
    """If rel_path not present in repo, try to download release asset filename into ROOT / rel_path."""
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
        else:
            return None
    except Exception:
        return None

# -----------------------
# Model loading
# -----------------------
@st.cache_resource
def load_models() -> Dict[str, Any]:
    model_specs = {
        "Logistic Regression": ("models/final_model_pipeline_lr.pkl", "final_model_pipeline_lr.pkl"),
        "LightGBM": ("models/final_model_pipeline_lgb.pkl", "final_model_pipeline_lgb.pkl"),
        "Random Forest": ("models/final_model_pipeline_rfr.pkl", "final_model_pipeline_rfr.pkl"),
        "XGBoost": ("models/final_model_pipeline_xgb.pkl", "final_model_pipeline_xgb.pkl"),
    }
    loaded = {}
    for name, (rel_path, release_asset) in model_specs.items():
        p = ROOT / rel_path
        if not p.exists():
            # attempt download from release asset
            download_asset_if_missing(rel_path, release_asset)
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
# Helpers: predict and safe extract proba
# -----------------------
def safe_predict(mdl, X: pd.DataFrame):
    """Return (preds_array, probs_array_or_None, error_or_None)"""
    try:
        preds = mdl.predict(X)
        preds = np.array(preds).ravel()
    except Exception as e:
        return None, None, e
    # try probability
    proba = None
    try:
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(X)
            # convert to failure-prob (assuming class 1 is failure)
            if proba.shape[1] >= 2:
                proba = np.array(proba)[:, 1]
            else:
                proba = None
        else:
            proba = None
    except Exception:
        proba = None
    return preds, proba, None

# -----------------------
# UI: Title + description
# -----------------------
st.title("⚙️ Machine Failure Prediction App")
st.markdown("""
Upload sensor CSV files for batch predictions **or** use the left sidebar for a single-sample prediction.
This app runs multiple saved pipelines (LR / LGB / RFR / XGB) and shows model-wise predictions + an ensemble.
""")

# -----------------------
# Sidebar: single-sample inputs
# -----------------------
st.sidebar.header("Input Features (single sample)")

air_temp_c = st.sidebar.number_input("Air Temperature (°C)", value=300.0, step=1.0, format="%.2f")
process_temp_c = st.sidebar.number_input("Process Temperature (°C)", value=305.0, step=1.0, format="%.2f")
rot_speed = st.sidebar.number_input("Rotational Speed (rpm)", value=1800.0, step=1.0)
torque = st.sidebar.number_input("Torque (Nm)", value=40.0, step=1.0)
tool_wear = st.sidebar.number_input("Tool Wear (min)", value=150.0, step=1.0)

model_choice = st.sidebar.selectbox("Choose model for single-sample (or 'Ensemble')",
                                    options=["Ensemble"] + list(models.keys()))

# -----------------------
# Batch upload options
# -----------------------
st.sidebar.markdown("---")
allow_downloads = st.sidebar.checkbox("Allow per-file CSV download (batch)", value=True)
show_head = st.sidebar.checkbox("Show head preview (batch)", value=True)

# -----------------------
# Prepare single-sample DataFrame
# -----------------------
def build_input_df(air_c, proc_c, rpm, tq, wear):
    # model pipelines expect Kelvin for temps in this repo
    return pd.DataFrame({
        "Air temperature [K]": [air_c + 273.15],
        "Process temperature [K]": [proc_c + 273.15],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [tq],
        "Tool wear [min]": [wear],
    })

single_df = build_input_df(air_temp_c, process_temp_c, rot_speed, torque, tool_wear)

# --------------------------------
# Left column: single-sample section
# --------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Single-sample prediction")
    st.write("Use the sidebar inputs, choose `Ensemble` or a model and press Predict.")
    if st.button("Predict Failure"):
        st.info("Running predictions...")
        # run each model
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

        # build ensemble average across available probability values; if probabilities missing, ensemble on preds
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

        # show results
        st.markdown("### Model outputs")
        rows = []
        for name, res in model_results.items():
            if "error" in res:
                rows.append({"Model": name, "Pred": "ERR", "Prob": "-", "Notes": res["error"]})
            else:
                rows.append({"Model": name, "Pred": int(res["pred"]), "Prob": f"{res['proba']:.3f}" if res['proba'] is not None else "-", "Notes": ""})
        df_results = pd.DataFrame(rows)
        st.table(df_results.set_index("Model"))

        st.markdown("### Ensemble result")
        if ensemble_pred is None:
            st.warning("No successful model outputs to produce an ensemble.")
        else:
            if ensemble_pred == 1:
                if ensemble_prob is not None:
                    st.error(f"🚨 Ensemble: MACHINE LIKELY TO FAIL (prob = {ensemble_prob:.2f})")
                else:
                    st.error(f"🚨 Ensemble: MACHINE LIKELY TO FAIL (by majority vote)")
            else:
                if ensemble_prob is not None:
                    st.success(f"✅ Ensemble: MACHINE SAFE (failure prob = {ensemble_prob:.2f})")
                else:
                    st.success("✅ Ensemble: MACHINE SAFE (by majority vote)")

            # show probability bar chart and gauge
            fig, ax = plt.subplots(figsize=(6, 2.2))
            model_names = []
            model_probs = []
            for name, res in model_results.items():
                if "prob" in res and res.get("prob") is not None:
                    pass
                # we stored under 'proba' earlier: normalize
            # re-fetch for plotting
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
                ax.barh(model_names, model_probs, height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Failure probability")
                ax.set_title("Model failure probabilities")
                for i, v in enumerate(model_probs):
                    ax.text(v + 0.01, i, f"{v:.2f}", va='center')
                st.pyplot(fig)
            else:
                st.info("No probability outputs available for bar chart (models may not expose predict_proba).")

        # feature importance attempt (use first model that has it)
        feat_fig = None
        for name, mdl in models.items():
            if isinstance(mdl, Exception):
                continue
            # try to resolve underlying estimator
            estimator = getattr(mdl, "named_steps", None)
            try:
                # If pipeline, attempt last step
                if hasattr(mdl, "named_steps"):
                    last = list(mdl.named_steps.values())[-1]
                else:
                    last = mdl
                if hasattr(last, "feature_importances_"):
                    fi = np.array(last.feature_importances_)
                    # feature names — if pipeline with transformer that outputs columns, attempt to get feature names from input columns
                    cols = list(single_df.columns)
                    # safe length check
                    if len(fi) == len(cols):
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        ax2.barh(cols, fi)
                        ax2.set_title(f"Feature importances — {name}")
                        st.pyplot(fig2)
                    break
            except Exception:
                continue

with col2:
    # -----------------------
    # Batch section: upload CSV(s)
    # -----------------------
    st.header("Batch prediction — CSV upload")
    st.write("Upload one or more CSV files with sensor features. App will run all available models for each file and allow per-file download of results.")
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

            # run models
            preds_dict = {}
            probs_dict = {}
            model_msgs = {}
            for name, mdl in models.items():
                if isinstance(mdl, Exception):
                    model_msgs[name] = str(mdl)
                    preds_dict[name] = np.full(len(df), np.nan)
                    probs_dict[name] = np.full(len(df), np.nan)
                    continue
                try:
                    preds, proba, err = safe_predict(mdl, df)
                    if err:
                        model_msgs[name] = str(err)
                        preds_dict[name] = np.full(len(df), np.nan)
                        probs_dict[name] = np.full(len(df), np.nan)
                    else:
                        preds_dict[name] = np.asarray(preds)
                        probs_dict[name] = np.asarray(proba) if proba is not None else np.full(len(df), np.nan)
                except Exception as e:
                    model_msgs[name] = str(e)
                    preds_dict[name] = np.full(len(df), np.nan)
                    probs_dict[name] = np.full(len(df), np.nan)

            # combine into result df
            result_df = pd.DataFrame(index=np.arange(len(df)))
            for name in models.keys():
                result_df[name + "_pred"] = preds_dict.get(name, np.full(len(df), np.nan))
                result_df[name + "_prob"] = probs_dict.get(name, np.full(len(df), np.nan))

            # ensemble average of probabilities where available
            prob_cols = [c for c in result_df.columns if c.endswith("_prob")]
            result_df["Ensemble_prob"] = result_df[prob_cols].mean(axis=1, skipna=True)
            # if probabilities mostly NaN, fallback to majority vote on _pred
            if result_df["Ensemble_prob"].isna().all():
                pred_cols = [c for c in result_df.columns if c.endswith("_pred")]
                result_df["Ensemble_vote"] = result_df[pred_cols].mode(axis=1, numeric_only=True)[0]
            else:
                result_df["Ensemble_pred"] = (result_df["Ensemble_prob"] >= 0.5).astype(int)

            # attach source file info and optionally an id column if original df had one
            if "id" in df.columns:
                result_df.insert(0, "id", df["id"].values)
            result_df.insert(0, "source_file", f.name)

            all_results.append(result_df)

            # allow per-file download
            if allow_downloads:
                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button(f"Download predictions for {f.name}", csv_bytes, file_name=f"{Path(f.name).stem}_predictions.csv")

            progress_bar.progress(idx / total)

        # combine and show
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            st.header("Combined predictions preview")
            st.dataframe(combined.head(30))
            st.download_button("Download combined CSV", combined.to_csv(index=False).encode(), "combined_predictions.csv")

# -----------------------
# Report generation (PDF)
# -----------------------
st.markdown("---")
st.header("Export a PDF report (single-sample)")

def create_pdf_report(single_df, model_results, ensemble_pred, ensemble_prob):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Machine Failure Prediction Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Input (single sample):", ln=True)
    for k, v in single_df.iloc[0].items():
        pdf.cell(0, 7, f" - {k}: {v}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, "Model outputs:", ln=True)
    for name, res in model_results.items():
        if "error" in res:
            pdf.cell(0, 6, f" - {name}: ERROR ({res['error']})", ln=True)
        else:
            pdf.cell(0, 6, f" - {name}: pred={res['pred']}, prob={res['proba']:.3f}" if res['proba'] is not None else f" - {name}: pred={res['pred']}", ln=True)
    pdf.ln(6)
    pdf.cell(0, 8, "Ensemble:", ln=True)
    if ensemble_pred is None:
        pdf.cell(0, 6, "No ensemble could be computed.", ln=True)
    else:
        pdf.cell(0, 6, f"Ensemble pred: {ensemble_pred}; prob: {ensemble_prob}", ln=True)

    # embed a quick probability bar png for the ensemble if possible
    try:
        # generate a matplotlib image like earlier
        fig, ax = plt.subplots(figsize=(4,1.2))
        names = []
        probs = []
        for name,res in model_results.items():
            if "error" in res: continue
            if res.get("proba") is not None:
                names.append(name)
                probs.append(res.get("proba"))
        if probs:
            ax.barh(names, probs, height=0.6)
            ax.set_xlim(0,1)
            ax.set_xlabel("Failure probability")
            ax.set_title("Model probabilities")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            pdf.image(buf, x=10, w=190)
    except Exception:
        pass

    out = io.BytesIO()
    out.write(pdf.output(dest='S').encode('latin1'))
    out.seek(0)
    return out

# Quick create report button handling
if st.button("Create & download PDF report (single sample)"):
    # re-run the single-sample predictions (like above)
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

    # ensemble calculation
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

    pdf_bytes = create_pdf_report(single_df, model_results, ensemble_pred, ensemble_prob)
    st.download_button("Download PDF report", pdf_bytes, file_name="machine_failure_report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("If a model file was missing the app attempted to download it from the repository releases. Check app logs in Streamlit Cloud if something failed. For best reliability, upload the `models/*.pkl` into the `models/` folder in the repo and redeploy.")
