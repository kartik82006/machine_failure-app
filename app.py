import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Machine Failure Prediction", layout="wide")

st.title("🧠 Machine Failure Prediction App")
st.write("Upload your dataset and run predictions using trained ML models.")

# -----------------------------------------------------------
# Helper: Load model safely
# -----------------------------------------------------------
def load_model(path):
    if not os.path.exists(path):
        return None, f"Missing model file: {path}"
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, "Loaded successfully"
    except Exception as e:
        return None, str(e)

# -----------------------------------------------------------
# File Upload
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df.head())

    # Automatically detect target = LAST COLUMN
    target_col = df.columns[-1]
    st.success(f"Detected target column: **{target_col}**")

    # Split input/output
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------------------------------------
    # Model files
    # -------------------------------------------------------
    model_paths = {
        "Logistic Regression": "models/final_model_pipeline_lr.pkl",
        "LightGBM": "models/final_model_pipeline_lgb.pkl",
        "Random Forest": "models/final_model_pipeline_rfr.pkl",
        "XGBoost": "models/final_model_pipeline_xgb.pkl"
    }

    # -------------------------------------------------------
    # Prediction Table Setup
    # -------------------------------------------------------
    results = []

    for model_name, path in model_paths.items():
        model, status = load_model(path)

        if model is None:
            results.append([model_name, "ERROR", "—", status])
        else:
            try:
                pred = model.predict(X)
                prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else "N/A"

                results.append([
                    model_name,
                    pred[0] if len(pred) > 0 else "N/A",
                    prob[0] if isinstance(prob, (list, pd.Series)) else prob,
                    "Success"
                ])
            except Exception as e:
                results.append([model_name, "ERROR", "—", f"Pipeline error: {str(e)}"])

    # -------------------------------------------------------
    # Display Results
    # -------------------------------------------------------
    st.subheader("📊 Model Predictions")
    results_df = pd.DataFrame(
        results,
        columns=["Model", "Prediction", "Prob", "Notes"]
    )
    st.dataframe(results_df, use_container_width=True)

else:
    st.info("Please upload a CSV file to continue.")
