import os
import json
import boto3
import io
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    roc_auc_score, accuracy_score
)

# Load AWS region, bucket, and Athena output path from config
config = json.load(open("config/config.json"))
AWS_REGION = config["region"]
ATHENA_OUTPUT_PREFIX = config["athena_output"].replace("s3://", "")
S3_BUCKET = ATHENA_OUTPUT_PREFIX.split("/")[0]
S3_KEY_PREFIX = "/".join(ATHENA_OUTPUT_PREFIX.split("/")[1:])

# Setting Local output and data directories
LOCAL_OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
LOCAL_DASHBOARD_PATH = os.path.join(LOCAL_OUTPUT_DIR, "olist_eda_dashboard.html")

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)
LOCAL_WRANGLED_CSV = os.path.join(DATA_DIR, "wrangled_sample.csv")

# Initializing Athena and S3 clients
s3_client = boto3.client("s3", region_name=AWS_REGION)
athena_client = boto3.client("athena", region_name=AWS_REGION)


# Function to convert the value to float, returns 0.0 if conversion fails
def safe_float(value):
    """Safely converts value to float; returns 0.0 if conversion fails."""
    try:
        return float(value)
    except Exception:
        return 0.0

# Function to fetch the wrangled dataset from Athena and save locally
def fetch_dataset_from_athena():    
    print(f"Exporting wrangled dataset from Athena ({config['catalog_database']}.wrangled)...")

    import awswrangler as wr
    database_name = config["catalog_database"]

    # SQL query to select relevant features for ML
    sql_query = f"""
    SELECT
      customer_state, seller_state,
      price, freight_value, product_weight_g,
      product_category_name_english,
      order_delivered_customer_date, order_estimated_delivery_date
    FROM {database_name}.wrangled
    WHERE order_delivered_customer_date IS NOT NULL
    """

    # Use AWS Wrangler to run the Athena query and export results
    df = wr.athena.read_sql_query(
        sql=sql_query,
        database=database_name,
        ctas_approach=False,
        workgroup="primary",
        s3_output=f"s3://{S3_BUCKET}/athena-results/",
        boto3_session=boto3.Session(region_name=AWS_REGION)
    )

    # Save locally for reuse
    os.makedirs(os.path.dirname(LOCAL_WRANGLED_CSV), exist_ok=True)
    df.to_csv(LOCAL_WRANGLED_CSV, index=False)
    print(f"Data exported locally → {LOCAL_WRANGLED_CSV}")
    return df

# Loading wrangled dataset from local CSV to Athena
print("\n")
print("Running ML: Late Delivery Prediction")


if os.path.exists(LOCAL_WRANGLED_CSV):
    wrangled_df = pd.read_csv(LOCAL_WRANGLED_CSV)
    print(f"Loaded existing dataset from {LOCAL_WRANGLED_CSV}")
else:
    try:
        import awswrangler as wr
        wrangled_df = fetch_dataset_from_athena()
    except Exception as error:
        print("Athena fetch failed with error", error)
        print("No data for ML model.")
        exit()

# Feature Engineering
print("Preparing features...")

# Calculate delay in days and create binary target variable (is_late)
wrangled_df["delay_days"] = (
    pd.to_datetime(wrangled_df["order_delivered_customer_date"])
    - pd.to_datetime(wrangled_df["order_estimated_delivery_date"])
).dt.days
wrangled_df["is_late"] = (wrangled_df["delay_days"] > 0).astype(int)

# Select only required columns
columns_needed = [
    "price", "freight_value", "product_weight_g",
    "customer_state", "seller_state",
    "product_category_name_english", "is_late"
]
wrangled_df = wrangled_df[[col for col in columns_needed if col in wrangled_df.columns]].dropna()

# Convert numeric fields to float
for numeric_col in ["price", "freight_value", "product_weight_g"]:
    wrangled_df[numeric_col] = wrangled_df[numeric_col].astype(float)

# Encode categorical features numerically
for cat_col in ["customer_state", "seller_state", "product_category_name_english"]:
    if cat_col in wrangled_df.columns:
        wrangled_df[cat_col] = LabelEncoder().fit_transform(wrangled_df[cat_col].astype(str))

# Split features and target
X = wrangled_df.drop(columns=["is_late"])
y = wrangled_df["is_late"]

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting Data into train and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Training logistic regression and random forest models
print("Training Logistic Regression and Random Forest models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, max_depth=10, n_jobs=-1
    ),
}
model_results = {}
for model_name, model_instance in models.items():
    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_test)
    y_pred_proba = model_instance.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    model_results[model_name] = {
        "y_pred": y_pred,
        "y_proba": y_pred_proba,
        "auc": auc_score,
        "acc": accuracy,
    }
    print(f"{model_name}: AUC={auc_score:.3f}, Accuracy={accuracy:.3f}")


# Building Evaluation Charts
print("Building ML comparison charts...")

# --- ROC Curve ---
roc_chart = go.Figure()
for model_name, metrics in model_results.items():
    fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
    roc_chart.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={metrics['auc']:.3f})")
    )
roc_chart.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                               name="Random", line=dict(dash="dash")))
roc_chart.update_layout(title="ROC Curve Comparison",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate")

# --- Feature Importance (Random Forest) ---
rf_importances = pd.Series(
    models["Random Forest"].feature_importances_, index=X.columns
).sort_values()
importance_chart = go.Figure([go.Bar(x=rf_importances.index, y=rf_importances.values)])
importance_chart.update_layout(title="Random Forest Feature Importance",
                               xaxis_title="Feature",
                               yaxis_title="Importance")

# --- Confusion Matrices ---
conf_matrix_lr = confusion_matrix(y_test, model_results["Logistic Regression"]["y_pred"])
conf_matrix_rf = confusion_matrix(y_test, model_results["Random Forest"]["y_pred"])

conf_matrix_chart = go.Figure()
conf_matrix_chart.add_trace(go.Heatmap(
    z=conf_matrix_lr, x=["Pred OnTime", "Pred Late"],
    y=["Actual OnTime", "Actual Late"],
    showscale=False, colorscale="Blues", name="LogReg"
))
conf_matrix_chart.add_trace(go.Heatmap(
    z=conf_matrix_rf, x=["Pred OnTime", "Pred Late"],
    y=["Actual OnTime", "Actual Late"],
    showscale=False, colorscale="Greens", name="RandForest"
))
conf_matrix_chart.update_layout(title="Confusion Matrices (LogReg=Blue, RF=Green)")

# Appending ML insights to existing HTML dashboard
print("Appending ML section to combined dashboard...")

ml_dashboard_section = f"""
<hr>
<div class="container my-4">
  <h2 class="text-center mb-3">🤖 Machine Learning Insights</h2>
  <p class="text-center text-muted">Late Delivery Prediction using Logistic Regression vs Random Forest</p>

  <div class="row text-center mb-4">
    <div class="col-md-3 mb-3"><div class="card shadow-sm border-0"><div class="card-body">
      <h6 class="text-muted">LogReg AUC</h6>
      <h4 class="fw-bold">{model_results['Logistic Regression']['auc']:.3f}</h4>
    </div></div></div>

    <div class="col-md-3 mb-3"><div class="card shadow-sm border-0"><div class="card-body">
      <h6 class="text-muted">RandForest AUC</h6>
      <h4 class="fw-bold">{model_results['Random Forest']['auc']:.3f}</h4>
    </div></div></div>

    <div class="col-md-3 mb-3"><div class="card shadow-sm border-0"><div class="card-body">
      <h6 class="text-muted">LogReg Accuracy</h6>
      <h4 class="fw-bold">{model_results['Logistic Regression']['acc']:.3f}</h4>
    </div></div></div>

    <div class="col-md-3 mb-3"><div class="card shadow-sm border-0"><div class="card-body">
      <h6 class="text-muted">RandForest Accuracy</h6>
      <h4 class="fw-bold">{model_results['Random Forest']['acc']:.3f}</h4>
    </div></div></div>
  </div>

  <div class="card shadow-sm mb-4"><div class="card-body">
    <h5>ROC Curve Comparison</h5>
    {roc_chart.to_html(full_html=False, include_plotlyjs="cdn")}
  </div></div>

  <div class="card shadow-sm mb-4"><div class="card-body">
    <h5>Feature Importance (Random Forest)</h5>
    {importance_chart.to_html(full_html=False, include_plotlyjs=False)}
  </div></div>

  <div class="card shadow-sm mb-4"><div class="card-body">
    <h5>Confusion Matrices</h5>
    {conf_matrix_chart.to_html(full_html=False, include_plotlyjs=False)}
  </div></div>
</div>
"""

# Append ML insights section into the existing dashboard
if os.path.exists(LOCAL_DASHBOARD_PATH):
    with open(LOCAL_DASHBOARD_PATH, "r+", encoding="utf-8") as html_file:
        dashboard_html = html_file.read()
        if "</body>" in dashboard_html:
            dashboard_html = dashboard_html.replace("</body>", ml_dashboard_section + "\n</body>")
        else:
            dashboard_html += ml_dashboard_section
        html_file.seek(0)
        html_file.write(dashboard_html)
        html_file.truncate()

print(f"ML comparison appended to {LOCAL_DASHBOARD_PATH}")
