# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
import io

st.set_page_config(page_title="Machine Failure — Auto Dashboard", layout="wide")
st.title("Machine Failure Prediction — Auto Dashboard")
st.markdown("This dashboard automatically loads CSV files and model pipelines from the repo (or from Releases if needed).")

ROOT = Path(__file__).parent

# ---------- Configure model filenames and release URLs ----------
# If you moved models to GitHub Releases, set the release URLs correctly here.
# If you keep models in repo under `models/` or root, the app will load them locally.
MODEL_FILES = {
    "LightGBM": {"local": ROOT / "models" / "final_model_pipeline_lgb.pkl", "release_url": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_lgb.pkl"},
    "Logistic Regression": {"local": ROOT / "models" / "final_model_pipeline_lr.pkl", "release_url": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_lr.pkl"},
    "Random Forest": {"local": ROOT / "models" / "final_model_pipeline_rfr.pkl", "release_url": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_rfr.pkl"},
    "XGBoost": {"local": ROOT / "models" / "final_model_pipeline_xgb.pkl", "release_url": "https://github.com/DURVA-GARGGG/machine_failure-app/releases/download/v1/final_model_pipeline_xgb.pkl"},
}

# Create models directory if not present (safe)
( ROOT / "models" ).mkdir(exist_ok=True)

def download_to(path: Path, url: str, chunk_size=8192):
    """Download a file and save it to path. Overwrites if exists."""
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)

@st.cache_resource
def load_models():
    models = {}
    for name, info in MODEL_FILES.items():
        p_local = info["local"]
        # 1) try local path
        if p_local.exists():
            try:
                models[name] = joblib.load(p_local)
                continue
            except Exception as e:
                models[name] = e
                continue
        # 2) try download from release URL
        url = info.get("release_url")
        if url:
            try:
                # download into models/<filename>
                target = p_local
                ok, err = download_to(target, url)
                if not ok:
                    models[name] = FileNotFoundError(f"Could not download {name}: {err}")
                    continue
                models[name] = joblib.load(target)
            except Exception as e:
                models[name] = e
        else:
            models[name] = FileNotFoundError(f"Missing model file for {name}")
    return models

models = load_models()

# ---------- Find CSV files in repo (root and dataset/ directory) ----------
def find_csv_files():
    candidates = []
    # look in repo root
    for p in ROOT.glob("*.csv"):
        candidates.append(p)
    # look in dataset/
    dataset_dir = ROOT / "dataset"
    if dataset_dir.exists():
        for p in dataset_dir.glob("*.csv"):
            candidates.append(p)
    # also look in data/ or dataset/*
    data_dir = ROOT / "data"
    if data_dir.exists():
        for p in data_dir.glob("*.csv"):
            candidates.append(p)
    # deduplicate and sort
    candidates = sorted({str(x): x for x in candidates}.values())
    return candidates

csv_paths = find_csv_files()
if not csv_paths:
    st.error("No CSV files found in repository root or dataset/ or data/. Please add your CSVs to the repo.")
    st.stop()

st.sidebar.header("Options")
selected_csv = st.sidebar.selectbox("Choose CSV to process", [p.name for p in csv_paths])
show_head = st.sidebar.checkbox("Show preview of source CSV", value=True)
show_model_details = st.sidebar.checkbox("Show per-model messages/errors", value=False)

# load chosen CSV
active_csv_path = next(p for p in csv_paths if p.name == selected_csv)
try:
    df_source = pd.read_csv(active_csv_path)
except Exception as e:
    st.error(f"Failed to read {active_csv_path.name}: {e}")
    st.stop()

st.subheader(f"Source file: {active_csv_path.name}")
if show_head:
    st.dataframe(df_source.head())

# ---------- Run predictions ----------
st.markdown("### Running models on the source data...")
results = {}
model_messages = {}

# Ensure df_source columns are what the model expects — we will pass full df and rely on pipeline preprocessing
for model_name, mdl in models.items():
    if isinstance(mdl, Exception):
        model_messages[model_name] = f"Load error: {mdl}"
        continue
    try:
        preds = mdl.predict(df_source)
        preds = np.array(preds).ravel()
        results[model_name] = preds
    except Exception as e:
        model_messages[model_name] = f"Predict error: {e}"

# Build a DataFrame of model outputs (align by row count)
n = len(df_source)
out_df = pd.DataFrame(index=np.arange(n))

for model_name in MODEL_FILES.keys():
    val = results.get(model_name)
    if val is None:
        out_df[model_name] = np.nan
    else:
        # ensure length matches source rows
        if len(val) == n:
            out_df[model_name] = val
        else:
            # mismatch: try broadcasting or fill with NaN
            try:
                out_df[model_name] = np.resize(val, n)
                model_messages[model_name] = model_messages.get(model_name, "") + " (prediction resized to match rows)"
            except Exception:
                out_df[model_name] = np.nan
                model_messages[model_name] = model_messages.get(model_name, "") + " (prediction length mismatch)"

# Ensemble: mean of numeric model columns (skips NaN)
numeric_cols = out_df.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    out_df["Ensemble_Average"] = numeric_cols.mean(axis=1)
else:
    out_df["Ensemble_Average"] = np.nan

# Attach id column if present
id_col = None
for candidate in ["id", "Id", "ID", "machine_id", "machineID"]:
    if candidate in df_source.columns:
        id_col = candidate
        break
if id_col:
    out_df.insert(0, id_col, df_source[id_col].values)

# show results
st.success("Predictions finished.")
st.subheader("Sample of prediction results")
st.dataframe(out_df.head(10))

# show summary metrics
st.markdown("### Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Rows processed", value=len(out_df))
    st.metric("Models used", value=len([k for k in models.keys()]))
with col2:
    # show simple ensemble distribution
    if "Ensemble_Average" in out_df.columns:
        st.write("Ensemble distribution (first 200 rows)")
        st.bar_chart(out_df["Ensemble_Average"].dropna().head(200))

# show model messages if any
if show_model_details:
    st.markdown("### Model load / predict messages")
    st.json({k: str(v) for k, v in model_messages.items()})

# download buttons
st.markdown("---")
csv_bytes = out_df.to_csv(index=False).encode()
st.download_button("Download predictions CSV", csv_bytes, "predictions.csv")

# Combined display: add original features + predictions optionally
if st.checkbox("Show combined data (source + predictions)", value=False):
    combined = pd.concat([df_source.reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)
    st.dataframe(combined.head(20))

st.caption("If some models failed to load, consider placing model .pkl files into a `models/` folder in the repo root, or upload them to GitHub Releases and update URLs in app.py.")
