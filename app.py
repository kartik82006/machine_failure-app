import streamlit as st
import pandas as pd
import pickle
import base64
import os

st.set_page_config(page_title="Machine Failure Prediction", layout="wide")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.info(
    "Upload a CSV file and the app will run predictions using all trained models."
)

# -------------------------------------------------------
# Helper: Load model safely
# -------------------------------------------------------
def load_model(path):
    if not os.path.exists(path):
        return None, f"❌ Missing file: {path}"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), "Loaded successfully"
    except Exception as e:
        return None, f"❌ Error loading: {str(e)}"

# -------------------------------------------------------
# File Upload Section
# -------------------------------------------------------
st.title("🧠 Machine Failure Prediction App")

uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Automatically detect last column as target
    target_col = df.columns[-1]
    st.success(f"Detected target column: **{target_col}**")

    # Prepare input features
    X = df.drop(columns=[target_col])

    # ---------------------------------------------------
    # Models dictionary
    # ---------------------------------------------------
    model_files = {
        "Logistic Regression": "final_model_pipeline_lr.pkl",
        "LightGBM": "final_model_pipeline_lgb.pkl",
        "Random Forest": "final_model_pipeline_rfr.pkl",
        "XGBoost": "final_model_pipeline_xgb.pkl",
    }

    results = []

    # ---------------------------------------------------
    # Predictions
    # ---------------------------------------------------
    st.subheader("📊 Predictions from All Models")

    for name, file in model_files.items():
        model, status = load_model(file)

        if model is None:
            results.append([name, "ERROR", "—", status])
        else:
            try:
                pred = model.predict(X)
                prob = (
                    model.predict_proba(X)[:, 1]
                    if hasattr(model, "predict_proba")
                    else ["N/A"] * len(pred)
                )

                results.append([
                    name,
                    pred[0] if len(pred) > 0 else "N/A",
                    prob[0] if isinstance(prob, (list, pd.Series)) else "N/A",
                    "✔️ Prediction successful"
                ])

            except Exception as e:
                results.append([name, "ERROR", "—", f"❌ {str(e)}"])

    # ---------------------------------------------------
    # Display results table
    # ---------------------------------------------------
    results_df = pd.DataFrame(
        results, columns=["Model", "Prediction", "Probability", "Status"]
    )
    st.dataframe(results_df, use_container_width=True)

    # ---------------------------------------------------
    # Download results as CSV or PDF
    # ---------------------------------------------------
    st.subheader("📥 Download Results")

    csv = results_df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "results.csv", "text/csv")

    # PDF Download using base64 HTML trick
    html = results_df.to_html(index=False)
    b64 = base64.b64encode(html.encode()).decode()
    pdf_link = f'<a href="data:application/octet-stream;base64,{b64}" download="results.html">Download PDF/HTML</a>'
    st.markdown(pdf_link, unsafe_allow_html=True)

else:
    st.info("Upload a CSV file to begin.")
