"""
Streamlit App - E-Commerce Furniture ML Pipeline

Features:
- Upload CSV (or use default file)
- Data cleaning & feature engineering
- Train pipelines: Linear Regression & RandomForest (GridSearchCV)
- Log-transform target, TF-IDF (1-2 grams), OneHotEncoder for tags
- Train/Test split (80/20) evaluation
- Learning curves, grid-search plot, feature importance
- Save pipelines (.joblib), Excel export, push to MySQL
"""

import os
import urllib.parse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import streamlit as st

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score

# sqlalchemy for MySQL
from sqlalchemy import create_engine, text

# ---------------------------
# Config / defaults
# ---------------------------
DEFAULT_CSV = "C:/Users/Hp/OneDrive/Desktop/python/ECommerce_Furniture_Dataset/ecommerce_furniture_dataset_2024.csv"
OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def clean_price_series(series):
    s = series.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
    s = pd.to_numeric(s.replace({'': np.nan, 'nan': np.nan}), errors='coerce')
    return s

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "r2": r2}

def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def get_feature_names_from_preprocessor(preprocessor):
    """Return combined feature names for ColumnTransformer preprocessor (after fit)."""
    # numeric
    num_feats = ["price", "discount_percentage"]
    cat_feats = []
    text_feats = []
    try:
        cat = preprocessor.named_transformers_["cat"]
        # OneHotEncoder -> get_feature_names_out
        cat_feats = list(cat.get_feature_names_out(["tagText_grouped"]))
    except Exception:
        cat_feats = []
    try:
        text = preprocessor.named_transformers_["text"]
        text_feats = list(text.get_feature_names_out())
    except Exception:
        text_feats = []
    return num_feats + cat_feats + text_feats

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="E-commerce Furniture ML", layout="wide")
st.title("E-commerce Furniture — ML pipeline (Streamlit)")

# Sidebar - data + DB creds
st.sidebar.header("Data / DB settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = False
if uploaded_file is None:
    if os.path.exists(DEFAULT_CSV):
        st.sidebar.write(f"No upload — using default CSV: `{DEFAULT_CSV}`")
        use_default = True
    else:
        st.sidebar.warning("No CSV uploaded and default not found. Please upload one.")
mysql_user = st.sidebar.text_input("MySQL user", value="root")
mysql_pass = st.sidebar.text_input("MySQL password", type="password")
mysql_host = st.sidebar.text_input("MySQL host", value="127.0.0.1")
mysql_port = st.sidebar.number_input("MySQL port", value=3306, step=1)
mysql_db = st.sidebar.text_input("MySQL database name", value="ecommerce_furniture")
tfidf_max_features = st.sidebar.number_input("TF-IDF max features", value=100, step=50)

# Buttons
train_button = st.sidebar.button("Train models (GridSearchCV)")
load_models_button = st.sidebar.button("Load saved pipelines (if available)")
export_excel_button = st.sidebar.button("Export predictions & metrics to Excel")
push_db_button = st.sidebar.button("Push metrics & predictions to MySQL")

# Main area - load data preview
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_default:
    df = pd.read_csv(DEFAULT_CSV)
else:
    st.stop()

st.subheader("Dataset preview")
st.write(f"Rows: {df.shape[0]}  —  Columns: {df.shape[1]}")
st.dataframe(df.head(6))

# Basic cleaning
st.subheader("Data cleaning & feature engineering")
with st.expander("Show cleaning code & steps"):
    st.markdown("""
- Clean `price`, `originalPrice` (remove $ and commas), convert to numeric
- Fill missing `productTitle` and `tagText`
- Compute `discount_percentage` from originalPrice when available
- `sold` converted to int
""")

# Apply cleaning
df.columns = [c.strip() for c in df.columns]
if "price" not in df.columns or "sold" not in df.columns or "productTitle" not in df.columns:
    st.error("CSV must contain columns: productTitle, price, sold. Aborting.")
    st.stop()

df["price"] = clean_price_series(df["price"])
if "originalPrice" in df.columns:
    df["originalPrice"] = clean_price_series(df["originalPrice"])
else:
    df["originalPrice"] = np.nan

df["sold"] = pd.to_numeric(df["sold"], errors="coerce").fillna(0).astype(int)
df["productTitle"] = df["productTitle"].fillna("").astype(str)
df["tagText"] = df["tagText"].fillna("unknown").astype(str)

df["discount_percentage"] = np.where(
    df["originalPrice"].notna() & (df["originalPrice"] > 0),
    ((df["originalPrice"] - df["price"]) / df["originalPrice"]) * 100,
    0.0
)

st.write("After cleaning (sample):")
st.dataframe(df[["productTitle", "price", "originalPrice", "discount_percentage", "sold", "tagText"]].head(6))

# Prepare variables that will be used later
y = df["sold"].values
y_log = np.log1p(y)

# Create top tags grouping
top_tags = df["tagText"].value_counts().nlargest(10).index
df["tagText_grouped"] = df["tagText"].apply(lambda x: x if x in top_tags else "others")

# Preprocessor (define, but will fit during training)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["price", "discount_percentage"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["tagText_grouped"]),
        ("text", TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,2), stop_words="english"), "productTitle"),
    ],
    remainder="drop"
)

# Local state to store trained objects
if "linreg_pipe" not in st.session_state:
    st.session_state["linreg_pipe"] = None
if "rf_pipe" not in st.session_state:
    st.session_state["rf_pipe"] = None
if "grid" not in st.session_state:
    st.session_state["grid"] = None

# Train models action
if train_button:
    st.sidebar.info("Training started — this may take a few minutes for GridSearchCV.")
    with st.spinner("Training..."):
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(df, y_log, test_size=0.2, random_state=RANDOM_STATE)

        # Ensure grouped tag column exists in split copies
        X_train["tagText_grouped"] = X_train["tagText"].apply(lambda x: x if x in top_tags else "others")
        X_test["tagText_grouped"] = X_test["tagText"].apply(lambda x: x if x in top_tags else "others")

        # Linear Regression pipeline
        linreg_pipe = Pipeline([("preprocess", preprocessor), ("regressor", LinearRegression())])
        linreg_pipe.fit(X_train, y_train)
        st.session_state["linreg_pipe"] = linreg_pipe

        # RandomForest pipeline + GridSearchCV
        rf_pipe = Pipeline([("preprocess", preprocessor), ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))])
        param_grid = {
            "regressor__n_estimators": [50],
            "regressor__max_depth": [10, None],
            "regressor__min_samples_split": [2, 5],
        }
        grid = GridSearchCV(rf_pipe, param_grid, cv=2, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best_rf = grid.best_estimator_
        st.session_state["rf_pipe"] = best_rf
        st.session_state["grid"] = grid

        # Predict on test set
        y_pred_lin = np.expm1(linreg_pipe.predict(X_test))
        y_pred_rf = np.expm1(best_rf.predict(X_test))
        y_test_orig = np.expm1(y_test)

        # Metrics
        metrics_lin = evaluate_model(y_test_orig, y_pred_lin)
        metrics_rf = evaluate_model(y_test_orig, y_pred_rf)

        metrics_df = pd.DataFrame([
            {"model": "LinearRegression", **metrics_lin},
            {"model": "RandomForest", **metrics_rf}
        ])
        metrics_df["generated_at"] = datetime.now().isoformat()
        st.session_state["metrics_df"] = metrics_df
        st.session_state["preds_df"] = pd.DataFrame({
            "productTitle": X_test["productTitle"].values,
            "true_sold": y_test_orig,
            "predicted_linreg": y_pred_lin,
            "predicted_rf": y_pred_rf,
            "price": X_test["price"].values,
            "discount_percentage": X_test["discount_percentage"].values,
            "tagText_grouped": X_test["tagText_grouped"].values
        }).reset_index(drop=True)

        # Save pipelines
        joblib.dump(linreg_pipe, os.path.join(OUTPUT_DIR, "linreg_pipeline.joblib"))
        joblib.dump(best_rf, os.path.join(OUTPUT_DIR, "rf_pipeline.joblib"))

        st.success("Training complete.")
        st.write("Best RF params:", grid.best_params_)

# Load models action
if load_models_button:
    lin_path = os.path.join(OUTPUT_DIR, "linreg_pipeline.joblib")
    rf_path = os.path.join(OUTPUT_DIR, "rf_pipeline.joblib")
    if os.path.exists(lin_path):
        st.session_state["linreg_pipe"] = joblib.load(lin_path)
        st.success("Loaded Linear Regression pipeline.")
    else:
        st.sidebar.warning("linreg_pipeline.joblib not found.")

    if os.path.exists(rf_path):
        st.session_state["rf_pipe"] = joblib.load(rf_path)
        st.success("Loaded Random Forest pipeline.")
    else:
        st.sidebar.warning("rf_pipeline.joblib not found.")

# If trained or loaded, show metrics and plots
if st.session_state.get("metrics_df") is not None:
    st.subheader("Metrics (on test set)")
    st.dataframe(st.session_state["metrics_df"])

    st.subheader("Predictions (test set sample)")
    st.dataframe(st.session_state["preds_df"].head(10))

    # Create & show plots
    # 1) Learning curves
    try:
        st.subheader("Learning curves")
        # For learning curve we need a model object (pipeline) that can be passed to learning_curve
        # Show for Linear Regression and Random Forest (best)
        X_train_full = df  # use full df for learning_curve input (preprocessor inside pipelines will fit)
        y_full = y_log

        fig1 = plt.figure(figsize=(7,4))
        train_sizes, train_scores, test_scores = learning_curve(
            st.session_state["linreg_pipe"], X_train_full, y_full,
            cv=3, n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5), scoring="neg_mean_squared_error"
        )
        train_mean = -np.mean(train_scores, axis=1)
        test_mean = -np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, "o-", label="Train MSE")
        plt.plot(train_sizes, test_mean, "o-", label="Val MSE")
        plt.title("Learning Curve - Linear Regression")
        plt.xlabel("Training size")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        lc_lin_path = os.path.join(PLOTS_DIR, "learning_curve_linreg.png")
        save_plot(fig1, lc_lin_path)
        st.image(lc_lin_path, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not produce learning curve for LinearRegression: {e}")

    try:
        fig2 = plt.figure(figsize=(7,4))
        train_sizes, train_scores, test_scores = learning_curve(
            st.session_state["rf_pipe"], X_train_full, y_full,
            cv=3, n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5), scoring="neg_mean_squared_error"
        )
        train_mean = -np.mean(train_scores, axis=1)
        test_mean = -np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, "o-", label="Train MSE")
        plt.plot(train_sizes, test_mean, "o-", label="Val MSE")
        plt.title("Learning Curve - Random Forest")
        plt.xlabel("Training size")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        lc_rf_path = os.path.join(PLOTS_DIR, "learning_curve_rf.png")
        save_plot(fig2, lc_rf_path)
        st.image(lc_rf_path, use_column_width=True)
    except Exception as e:
        st.warning(f"Could not produce learning curve for RandomForest: {e}")

    # 2) GridSearchCV results plot
    if st.session_state.get("grid") is not None:
        try:
            st.subheader("GridSearchCV results (Random Forest)")
            cv_results = pd.DataFrame(st.session_state["grid"].cv_results_)
            # Plot n_estimators vs mean_test_score (MSE)
            param_vals = cv_results["param_regressor__n_estimators"].astype(int)
            mean_test_mse = -cv_results["mean_test_score"]
            fig3 = plt.figure(figsize=(7,4))
            plt.plot(param_vals, mean_test_mse, "o-")
            plt.xlabel("n_estimators")
            plt.ylabel("Mean CV MSE")
            plt.title("Random Forest - GridSearchCV results")
            plt.grid(True)
            gs_path = os.path.join(PLOTS_DIR, "rf_gridsearch_results.png")
            save_plot(fig3, gs_path)
            st.image(gs_path, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not plot GridSearchCV results: {e}")

    # 3) Feature importance
    try:
        st.subheader("Random Forest - Feature importances (top 20)")
        pre = st.session_state["rf_pipe"].named_steps["preprocess"]
        feature_names = get_feature_names_from_preprocessor(pre)
        rf_est = st.session_state["rf_pipe"].named_steps["regressor"]
        importances = rf_est.feature_importances_
        # Align length
        if len(feature_names) == len(importances):
            feat_imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            feat_imp_df = feat_imp_df.sort_values("importance", ascending=False).head(20)
            fig4 = plt.figure(figsize=(8,6))
            plt.barh(feat_imp_df["feature"], feat_imp_df["importance"])
            plt.gca().invert_yaxis()
            plt.title("Top 20 Feature Importances - Random Forest")
            fi_path = os.path.join(PLOTS_DIR, "rf_feature_importance.png")
            save_plot(fig4, fi_path)
            st.image(fi_path, use_column_width=True)
            # save csv
            feat_imp_df.to_csv(os.path.join(OUTPUT_DIR, "rf_feature_importances.csv"), index=False)
            st.write(feat_imp_df)
        else:
            st.warning("Feature names length mismatch — cannot show feature importances.")
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

# Export to Excel
if export_excel_button:
    if st.session_state.get("metrics_df") is None or st.session_state.get("preds_df") is None:
        st.warning("No metrics/predictions in memory. Train models first or load pipelines.")
    else:
        excel_path = os.path.join(OUTPUT_DIR, f"predictions_and_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            st.session_state["metrics_df"].to_excel(writer, sheet_name="metrics_summary", index=False)
            st.session_state["preds_df"].to_excel(writer, sheet_name="predictions", index=False)
        st.success(f"Excel exported: {excel_path}")
        with open(excel_path, "rb") as f:
            st.download_button("Download Excel results", f, file_name=os.path.basename(excel_path))

# Push to MySQL
if push_db_button:
    if st.session_state.get("metrics_df") is None or st.session_state.get("preds_df") is None:
        st.warning("No metrics/predictions in memory. Train models first or load pipelines.")
    else:
        try:
            password_enc = urllib.parse.quote_plus(mysql_pass)
            engine_root_url = f"mysql+mysqlconnector://{mysql_user}:{password_enc}@{mysql_host}:{mysql_port}/"
            engine_root = create_engine(engine_root_url, echo=False, pool_recycle=3600)
            create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{mysql_db}`;"
            with engine_root.connect() as conn:
                conn.execute(text(create_db_sql))
            engine_url = f"mysql+mysqlconnector://{mysql_user}:{password_enc}@{mysql_host}:{mysql_port}/{mysql_db}"
            engine = create_engine(engine_url, echo=False, pool_recycle=3600)
            # Write metrics & preds
            st.session_state["metrics_df"].to_sql("model_metrics", engine, if_exists="replace", index=False)
            st.session_state["preds_df"].to_sql("model_predictions", engine, if_exists="replace", index=False)
            st.success(f"Wrote metrics and predictions to MySQL DB `{mysql_db}`.")
        except Exception as e:
            st.error(f"MySQL error: {e}")

# Footer — provide artifacts for download if exist
st.markdown("---")
st.subheader("Artifacts")
cols = st.columns(3)
with cols[0]:
    lin_path = os.path.join(OUTPUT_DIR, "linreg_pipeline.joblib")
    if os.path.exists(lin_path):
        with open(lin_path, "rb") as f:
            st.download_button("Download linreg_pipeline.joblib", f, file_name="linreg_pipeline.joblib")
with cols[1]:
    rf_path = os.path.join(OUTPUT_DIR, "rf_pipeline.joblib")
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            st.download_button("Download rf_pipeline.joblib", f, file_name="rf_pipeline.joblib")
with cols[2]:
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics_summary.csv")
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            st.download_button("Download metrics CSV", f, file_name="model_metrics_summary.csv")

st.caption("Tip: If GridSearchCV is slow, reduce TF-IDF features or grid size (in the sidebar).")
