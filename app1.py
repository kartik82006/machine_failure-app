# app1.py
from __future__ import annotations
import io
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests

import streamlit as st


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


st.set_page_config(page_title="Machine Failure — Ensemble", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).parent
GITHUB_RAW_RELEASE_BASE = "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/models"

# ---------- Toy fallback model (no sklearn needed) ----------
class ToyModel:
    """
    Lightweight fallback model that implements predict and predict_proba.
    Produces deterministic probabilities from a simple linear heuristic on the expected columns.
    """
    def __init__(self, name: str):
        self.name = name
        self.is_fallback = True

    def _ensure_df(self, X):
        # Accept DataFrame or array-like: if not DataFrame, try to convert
        if isinstance(X, pd.DataFrame):
            return X
        try:
            return pd.DataFrame(X)
        except Exception:
            raise ValueError("ToyModel expects a DataFrame or convertible input.")

    def _score(self, df: pd.DataFrame):
        # Use the 5 expected columns (if present) to create a weighted score (0..1)
        # We try to be robust if columns missing - use whatever available.
        cols = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
        vals = []
        for c in cols:
            if c in df.columns:
                # normalize roughly by a reasonable scale
                v = df[c].astype(float).fillna(0).values
                if "temperature" in c.lower():
                    # divide by 1000 so typical Kelvin ~ 500 -> becomes 0.5
                    vals.append(v / 1000.0)
                elif "rotational" in c.lower():
                    vals.append(v / 5000.0)
                elif "torque" in c.lower():
                    vals.append(v / 100.0)
                else:
                    vals.append(v / 300.0)
            else:
                vals.append(np.zeros(len(df)))
        # weighted sum
        weights = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
        stacked = np.vstack(vals)  # shape (5, n)
        raw = weights.reshape(-1, 1).T.dot(stacked)  # shape (1, n)
        raw = raw.ravel()
        # sigmoid to map to (0,1)
        probs = 1 / (1 + np.exp(-5*(raw - 0.5)))  # sharpen around 0.5
        probs = np.clip(probs, 0.001, 0.999)
        return probs

    def predict_proba(self, X):
        df = self._ensure_df(X)
        p = self._score(df)
        # return shape (n, 2) with columns [prob_0, prob_1]
        probs = np.vstack([1-p, p]).T
        return probs

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

# ---------- Helpers: download release asset if missing ----------
def download_asset_if_missing(rel_path: str, release_filename: str) -> Optional[Path]:
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

# ---------- Load models (cached) ----------
@st.cache_resource
def load_models() -> Dict[str, Any]:
    specs = {
        "Logistic Regression": ("final_model_pipeline_lr.pkl", "final_model_pipeline_lr.pkl"),
        "LightGBM": ("final_model_pipeline_lgb.pkl", "final_model_pipeline_lgb.pkl"),
        "Random Forest": ("final_model_pipeline_rfr.pkl", "final_model_pipeline_rfr.pkl"),
        "XGBoost": ("final_model_pipeline_xgb.pkl", "final_model_pipeline_xgb.pkl"),
    }
    loaded: Dict[str, Any] = {}
    for name, (rel_path, asset_name) in specs.items():
        p = ROOT / rel_path
        if not p.exists():
            # try download from release (best-effort)
            download_asset_if_missing(rel_path, asset_name)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    mdl = pickle.load(f)
                # quick sanity: check predict exists
                if not hasattr(mdl, "predict"):
                    raise ValueError("Loaded object has no predict()")
                loaded[name] = mdl
            except Exception as e:
                # fallback to toy but keep error message in notes
                loaded[name] = {"fallback": True, "error": str(e), "toy": ToyModel(name)}
        else:
            # missing file -> fallback toy model
            loaded[name] = {"fallback": True, "error": f"Missing model file: {rel_path}", "toy": ToyModel(name)}
    return loaded

_models_raw = load_models()

# Convert _models_raw to a uniform dict of model objects + note
# We'll create a dict mapping name -> (model_obj, note)
models: Dict[str, Dict[str, Any]] = {}
for name, entry in _models_raw.items():
    if isinstance(entry, dict) and entry.get("fallback"):
        models[name] = {"model": entry["toy"], "note": entry.get("error", "Using fallback toy model")}
    else:
        models[name] = {"model": entry, "note": None}

# ---------- Safe predict wrapper ----------
def safe_predict_wrapper(mdl_entry, X: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    mdl_entry is the dict stored in models[name] -> {'model': obj, 'note': maybe_string}
    Returns (preds, prob, note)
    - If model is a fallback toy, we return its outputs and the note.
    - If real model raises errors, return None and the error message as note.
    """
    note = mdl_entry.get("note")
    mdl = mdl_entry["model"]
    try:
        preds, proba, err = None, None, None
        if hasattr(mdl, "predict"):
            preds = mdl.predict(X)
        else:
            raise ValueError("Model object has no predict()")
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(X)
            if getattr(p, "ndim", 0) == 2 and p.shape[1] >= 2:
                proba = np.array(p)[:, 1]
            else:
                proba = None
        return np.array(preds).ravel(), (np.array(proba).ravel() if proba is not None else None), note
    except Exception as e:
        # if model is toy it shouldn't fail; otherwise return error note
        return None, None, str(e) if note is None else note + " | " + str(e)

# ---------- CSV utils ----------
REQUIRED_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

def read_csv_bytes_or_file(filelike) -> pd.DataFrame:
    if isinstance(filelike, (str, Path)):
        return pd.read_csv(str(filelike), encoding="utf-8-sig")
    else:
        return pd.read_csv(filelike, encoding="utf-8-sig")

def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0), missing

# ---------- UI ----------
st.title("⚙️ Machine Failure Prediction App")
st.markdown("Upload CSVs for batch predictions or use the sidebar for single-sample. Fallback toy models used when real pipelines missing or broken. ✅")

# Sidebar single-sample
st.sidebar.header("Input Features (single sample)")
air_temp_c = st.sidebar.number_input("Air Temperature (°C)", value=300.0, step=1.0, format="%.2f")
process_temp_c = st.sidebar.number_input("Process Temperature (°C)", value=305.0, step=1.0, format="%.2f")
rot_speed = st.sidebar.number_input("Rotational Speed (rpm)", value=1800.0, step=1.0)
torque = st.sidebar.number_input("Torque (Nm)", value=40.0, step=1.0)
tool_wear = st.sidebar.number_input("Tool Wear (min)", value=150.0, step=1.0)

st.sidebar.markdown("---")
allow_downloads = st.sidebar.checkbox("Allow per-file CSV download (batch)", value=True)
show_head = st.sidebar.checkbox("Show head preview (batch)", value=True)
st.sidebar.caption("Add 'newcsv.csv' at repo root or data/ to enable demo.")

def build_input_df(air_c, proc_c, rpm, tq, wear) -> pd.DataFrame:
    df = pd.DataFrame({
        "Type": ["M"],
        "Air temperature [K]": [300 + 273.15],
        "Process temperature [K]": [305 + 273.15],
        "Rotational speed [rpm]": [1800],
        "Torque [Nm]": [40],
        "Tool wear [min]": [150],
    })
    df["del_T"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power proxy"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    df["wear_rate"] = df["Tool wear [min]"] / df["Rotational speed [rpm]"]
    return df

single_df = build_input_df(air_temp_c, process_temp_c, rot_speed, torque, tool_wear)

# ---------- run batch ----------
def run_batch_on_df(df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
    ok, missing = validate_columns(df)
    if not ok:
        st.error(f"File `{source_name}` missing columns: {missing}. Use the example template.")
        return None

    if show_head:
        st.write(f"Preview — `{source_name}`")
        st.dataframe(df.head(5))

    preds_dict = {}
    probs_dict = {}
    notes_dict = {}

    for name, entry in models.items():
        preds, proba, note = safe_predict_wrapper(entry, df)
        if preds is None:
            preds_dict[name] = np.full(len(df), np.nan)
            probs_dict[name] = np.full(len(df), np.nan)
        else:
            preds_dict[name] = np.asarray(preds)
            probs_dict[name] = np.asarray(proba) if proba is not None else np.full(len(df), np.nan)
        notes_dict[name] = note or ""

    result_df = pd.DataFrame(index=np.arange(len(df)))
    for name in models.keys():
        result_df[name + "_pred"] = preds_dict.get(name, np.full(len(df), np.nan))
        result_df[name + "_prob"] = probs_dict.get(name, np.full(len(df), np.nan))

    prob_cols = [c for c in result_df.columns if c.endswith("_prob")]
    result_df["Ensemble_prob"] = result_df[prob_cols].mean(axis=1, skipna=True)

    if result_df["Ensemble_prob"].isna().all():
        pred_cols = [c for c in result_df.columns if c.endswith("_pred")]
        if pred_cols:
            modes = result_df[pred_cols].mode(axis=1, numeric_only=True)
            if not modes.empty:
                result_df["Ensemble_vote"] = modes.iloc[:, 0]
            else:
                result_df["Ensemble_vote"] = np.nan
        else:
            result_df["Ensemble_vote"] = np.nan
    else:
        result_df["Ensemble_pred"] = (result_df["Ensemble_prob"] >= 0.5).astype(int)

    if "id" in df.columns:
        result_df.insert(0, "id", df["id"].values)
    result_df.insert(0, "source_file", source_name)

    # small table of model notes for this file (helps to see fallback)
    notes_rows = [{"Model": name, "Notes": notes_dict.get(name, "")} for name in models.keys()]
    st.markdown("**Model Notes**")
    st.table(pd.DataFrame(notes_rows).set_index("Model"))

    st.success(f"Predictions generated for `{source_name}` ✅")
    st.dataframe(result_df.head(10))
    if allow_downloads:
        csv_bytes = result_df.to_csv(index=False).encode()
        st.download_button(f"Download predictions for {source_name}", csv_bytes, file_name=f"{Path(source_name).stem}_predictions.csv")

    return result_df

# ---------- Layout ----------
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Single-sample prediction")
    st.write("Fill sidebar values and press Predict.")

    # Create a unique key for tracking parameter changes
    current_params = (air_temp_c, process_temp_c, rot_speed, torque, tool_wear)
    
    # Initialize session state for predictions
    if 'predicted' not in st.session_state:
        st.session_state.predicted = False
    if 'last_params' not in st.session_state:
        st.session_state.last_params = None
    
    # Check if parameters have changed
    if st.session_state.last_params != current_params:
        st.session_state.predicted = False
        st.session_state.last_params = current_params

    if not st.session_state.predicted:
        if st.button("Predict Failure"):
            st.session_state.predicted = True
            st.rerun()
    
    if st.session_state.predicted:
        st.success("Predicted Failure ✅")
        
        # Map models to their corresponding failure type labels (matching notebook)
        label_to_model = {
            "TWF": "XGBoost",        # TWF: xgb (from notebook)
            "HDF": "LightGBM",       # HDF: lgb (from notebook)
            "PWF": "Random Forest",  # PWF: rfr (from notebook)
            "OSF": "XGBoost",        # OSF: xgb (from notebook)
            "RNF": "Logistic Regression"  # RNF: lr (from notebook)
        }
        
        # Load the actual trained models if they exist
        model_files = {
            "TWF": "final_model_pipeline_TWF_xgb.pkl",
            "HDF": "final_model_pipeline_HDF_lgb.pkl",
            "PWF": "final_model_pipeline_PWF_rfr.pkl",
            "OSF": "final_model_pipeline_OSF_xgb.pkl",
            "RNF": "final_model_pipeline_RNF_lr.pkl"
        }
        
        label_order = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        output_dict = {}
        label_results = {}
        
        for label in label_order:
            model_file = ROOT / "models" / model_files[label]
            
            # Try to load the specific trained model first
            if model_file.exists():
                try:
                    import joblib
                    loaded_model = joblib.load(model_file)
                    preds = loaded_model.predict(single_df)
                    proba = loaded_model.predict_proba(single_df)[:, 1] if hasattr(loaded_model, 'predict_proba') else None
                    note = ""
                except Exception as e:
                    # Fallback to generic models
                    model_name = label_to_model[label]
                    entry = models.get(model_name)
                    if entry is None:
                        entry = {"model": ToyModel(model_name), "note": f"Missing model file: {model_name}"}
                    preds, proba, note = safe_predict_wrapper(entry, single_df)
            else:
                # Fallback to generic models
                model_name = label_to_model[label]
                entry = models.get(model_name)
                if entry is None:
                    entry = {"model": ToyModel(model_name), "note": f"Missing model file: {model_name}"}
                preds, proba, note = safe_predict_wrapper(entry, single_df)
            
            if preds is None:
                output_dict[label] = "ERROR"
                label_results[label] = {"Prediction": "ERROR", "Probability": "—", "Notes": note}
            else:
                output_dict[label] = int(preds[0])
                label_results[label] = {
                    "Prediction": int(preds[0]),
                    "Probability": f"{float(proba[0]):.3f}" if proba is not None else "—",
                    "Notes": note or ""
                }
        
        # Display detailed table by label
        rows = []
        for label in label_order:
            res = label_results[label]
            rows.append({"Failure Type": label, **res})
        
        st.markdown("### Detailed Failure Type Predictions")
        st.dataframe(pd.DataFrame(rows).set_index("Failure Type"), use_container_width=True, height=250)

        # Chart for probabilities by label
        if _have_matplotlib:
            names = [row["Failure Type"] for row in rows if row["Probability"] not in ["—", "N/A"]]
            probs_plot = [float(row["Probability"]) for row in rows if row["Probability"] not in ["—", "N/A"]]
            if probs_plot:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(names, probs_plot, height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Failure probability")
                ax.set_title("Failure probabilities by type")
                for i, v in enumerate(probs_plot):
                    ax.text(v + 0.01, i, f"{v:.2f}", va="center")
                st.pyplot(fig)
            else:
                st.info("No probability outputs available for chart.")
        else:
            st.info("Plotting disabled (matplotlib missing).")

with col2:
    st.header("Batch prediction — CSV upload")
    st.write("Upload CSV files or run demo `newcsv.csv` (place at repo root or data/).")

    # Demo support
    demo_paths = [ROOT / "newcsv.csv", ROOT / "data" / "newcsv.csv"]
    demo_found = None
    for p in demo_paths:
        if p.exists():
            demo_found = p
            break

    if demo_found:
        st.success(f"Demo file found: `{demo_found.name}`")
        if st.button("Run demo on newcsv.csv"):
            try:
                demo_df = read_csv_bytes_or_file(demo_found)
            except Exception as e:
                st.error(f"Failed to read demo CSV `{demo_found}`: {e}")
                demo_df = None
            if demo_df is not None:
                run_batch_on_df(demo_df, str(demo_found.name))
    else:
        st.info("No demo CSV found in repo. Upload `newcsv.csv` to repo root or `data/`.")

    uploaded_files = st.file_uploader("Upload CSV files (accepts multiple)", type=["csv"], accept_multiple_files=True)
    all_results = []
    bad_files = []
    if uploaded_files:
        prog = st.progress(0)
        total = len(uploaded_files)
        for idx, f in enumerate(uploaded_files, start=1):
            st.subheader(f.name)

            try:
                df = pd.read_csv(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                bad_files.append(f.name)
                continue
            res = run_batch_on_df(df, f.name)
            if res is not None:
                all_results.append(res)
            prog.progress(idx / total)
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            st.header("Combined predictions preview")
            st.dataframe(combined.head(30))
            st.download_button("Download combined CSV", combined.to_csv(index=False).encode(), "combined_predictions.csv")

# ---------- PDF export ----------
st.markdown("---")
st.header("Export a PDF report (single-sample)")
if _have_fpdf:
    st.write("PDF export available.")
else:
    st.info("PDF export disabled (fpdf missing).")

if st.button("Create & download PDF report (single sample)"):
    # Re-run single-sample predictions for the report with same logic as main prediction
    label_to_model = {
        "TWF": "XGBoost",
        "HDF": "LightGBM",
        "PWF": "Random Forest",
        "OSF": "XGBoost",
        "RNF": "Logistic Regression"
    }
    
    model_files = {
        "TWF": "final_model_pipeline_TWF_xgb.pkl",
        "HDF": "final_model_pipeline_HDF_lgb.pkl",
        "PWF": "final_model_pipeline_PWF_rfr.pkl",
        "OSF": "final_model_pipeline_OSF_xgb.pkl",
        "RNF": "final_model_pipeline_RNF_lr.pkl"
    }
    
    label_order = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    label_results = {}
    
    for label in label_order:
        model_file = ROOT / "models" / model_files[label]
        
        # Try to load the specific trained model first
        if model_file.exists():
            try:
                import joblib
                loaded_model = joblib.load(model_file)
                preds = loaded_model.predict(single_df)
                proba = loaded_model.predict_proba(single_df)[:, 1] if hasattr(loaded_model, 'predict_proba') else None
                note = ""
            except Exception as e:
                # Fallback to generic models
                model_name = label_to_model[label]
                entry = models.get(model_name)
                if entry is None:
                    entry = {"model": ToyModel(model_name), "note": f"Missing model file: {model_name}"}
                preds, proba, note = safe_predict_wrapper(entry, single_df)
        else:
            # Fallback to generic models
            model_name = label_to_model[label]
            entry = models.get(model_name)
            if entry is None:
                entry = {"model": ToyModel(model_name), "note": f"Missing model file: {model_name}"}
            preds, proba, note = safe_predict_wrapper(entry, single_df)
        
        if preds is None:
            label_results[label] = {"pred": "ERROR", "proba": None, "note": note}
        else:
            label_results[label] = {
                "pred": int(preds[0]),
                "proba": float(proba[0]) if proba is not None else None,
                "note": note or ""
            }

    if not _have_fpdf:
        st.error("FPDF not installed; can't create PDF.")
    else:
        # create pdf
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Machine Failure Prediction Report", ln=True, align='C')
        pdf.ln(6)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "Input Parameters:", ln=True)
        pdf.cell(0, 7, f" - Air Temperature: {air_temp_c} C", ln=True)
        pdf.cell(0, 7, f" - Process Temperature: {process_temp_c} C", ln=True)
        pdf.cell(0, 7, f" - Rotational Speed: {rot_speed} rpm", ln=True)
        pdf.cell(0, 7, f" - Torque: {torque} Nm", ln=True)
        pdf.cell(0, 7, f" - Tool Wear: {tool_wear} min", ln=True)
        pdf.ln(4)
        pdf.cell(0, 8, "Failure Type Predictions:", ln=True)
        for label in label_order:
            res = label_results[label]
            if res["pred"] == "ERROR":
                pdf.cell(0, 6, f" - {label}: ERROR ({res['note']})", ln=True)
            else:
                if res["proba"] is not None:
                    pdf.cell(0, 6, f" - {label}: Prediction={res['pred']}, Probability={res['proba']:.3f}", ln=True)
                else:
                    pdf.cell(0, 6, f" - {label}: Prediction={res['pred']}", ln=True)

        out = io.BytesIO()
        out.write(pdf.output(dest='S').encode('latin1'))
        out.seek(0)
        st.download_button("Download PDF report", out, file_name="machine_failure_report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("If a real model file is missing the app uses a small deterministic ToyModel so you can demo the UI. For proper predictions deploy trained pipelines in models/*.pkl")