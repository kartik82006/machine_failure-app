import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

st.set_page_config(page_title="Machine Failure Prediction", layout="wide")


# -----------------------------
# Load Models Safely
# -----------------------------
@st.cache_resource
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None


models = {
    "Logistic Regression": load_model("final_model_pipeline_lr.pkl"),
    "Random Forest": load_model("final_model_pipeline_rfr.pkl"),
    "XGBoost": load_model("final_model_pipeline_xgb.pkl"),
    "LightGBM": load_model("final_model_pipeline_lgb.pkl"),
}


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Machine Failure Prediction")
st.sidebar.info(
    "Upload a CSV or use the demo input.\n\n"
    "All 4 models will run and generate predictions."
)


# -----------------------------
# Helper Functions
# -----------------------------
def prepare_features(df):
    """Drop the target column if present."""
    df = df.copy()
    if "Tumor" in df.columns:
        df = df.drop(columns=["Tumor"])
    return df


def predict_with_model(model, X):
    """Return prediction + probability or errors."""
    if model is None:
        return "ERROR", None, "Model file missing"

    try:
        pred = model.predict(X)[0]
        try:
            prob = model.predict_proba(X)[0][1]
        except:
            prob = None
        return pred, prob, "OK"
    except Exception as e:
        return "ERROR", None, "Pipeline is not fitted or input invalid"


def generate_pdf(result_df):
    """Generate PDF from DataFrame."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    textobject = c.beginText(40, 750)
    textobject.setFont("Helvetica", 10)

    for line in result_df.to_string().split("\n"):
        textobject.textLine(line)

    c.drawText(textobject)
    c.save()
    buffer.seek(0)
    return buffer


# -----------------------------
# MAIN UI
# -----------------------------
st.title("🧠 Machine Failure Prediction System")
st.write("Run all 4 ML models and compare predictions.")


# -----------------------------
# DEMO INPUT
# -----------------------------
if st.button("Use Demo Input"):
    st.session_state["input_df"] = pd.DataFrame(
        {
            "Air temperature [K]": [300],
            "Process temperature [K]": [310],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40],
            "Tool wear [min]": [120],
            "Type": ["L"]
        }
    )

# -----------------------------
# CSV UPLOAD
# -----------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df_input = pd.read_csv(uploaded)
elif "input_df" in st.session_state:
    df_input = st.session_state["input_df"]
else:
    st.stop()

st.subheader("🔍 Input Data")
st.dataframe(df_input)

X = prepare_features(df_input)

# -----------------------------
# RUN MODELS
# -----------------------------
results = []

for model_name, model in models.items():
    pred, prob, note = predict_with_model(model, X)
    results.append([model_name, pred, prob, note])

result_df = pd.DataFrame(results, columns=["Model", "Prediction", "Probability", "Notes"])

st.subheader("📊 Model Predictions")
st.dataframe(result_df, use_container_width=True)


# -----------------------------
# DOWNLOAD PDF
# -----------------------------
pdf_buffer = generate_pdf(result_df)

st.download_button(
    label="📥 Download Results as PDF",
    data=pdf_buffer,
    file_name="prediction_results.pdf",
    mime="application/pdf"
)
