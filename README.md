# 🛒 E-Commerce Sales Analytics – AWS ETL + EDA + ML Pipeline

### 📊 IIT Jodhpur Capstone Project – Group G16  
*End-to-End AWS Data Engineering + Machine Learning Pipeline on the E-Commerce Sales Dataset*

---

## 🏗️ System Architecture


<p align="center">
  <img width="512" height="768" alt="ArchitectureImage" src="https://github.com/user-attachments/assets/edced33e-f614-4b73-b92f-4b88a4e37a4e" />
</p>
**Figure:** High-level AWS architecture showing S3 data lake, Glue ETL jobs, Athena analytics,  
and ML dashboard generation pipeline.

---

## 🧭 Pipeline Overview

The project automates **data ingestion → transformation → analysis → prediction → dashboarding**  
using **AWS Glue, Athena, and S3**, integrated with **Python + PySpark + scikit-learn**.

### 🔁 Data & ML Flow

```text
Raw CSV (Denormalized)
        │
        ▼
[Glue Job 1] 🧩 Normalize → Split tables (orders, customers, etc.)
        │
        ▼
[Glue Job 2] 🔍 Understand → Inspect schemas & counts
        │
        ▼
[Glue Job 3] 🧹 Clean → Fix timestamps, nulls, typos, invalid values
        │
        ▼
[Glue Job 4] 🔗 Wrangle → Join all entities → "commerce" dataset
        │
        ▼
[Glue Crawler] 📚 Catalog tables in AWS Glue → Queryable in Athena
        │
        ▼
[Athena EDA] 📈 Run SQL analytics on wrangled dataset
        │
        ▼
[HTML Dashboard] 🧱 Build interactive Plotly + Bootstrap EDA dashboard
        │
        ▼
[ML Models] 🤖 Logistic Regression + Random Forest → Late delivery prediction
        │
        ▼
[Enhanced ML] 🚀 Gradient Boosting + SVC + extra metrics + feature insights
        │
        ▼
📊 Two Dashboards:
   • `olist_eda_dashboard.html` → Exploratory Data Analysis  
   • `olist_ml_dashboard.html` → Machine Learning Insights
```

---

## 💡 Project Summary

This project analyzes **E-Commerce sales data** to understand and predict **delivery performance**.  
It extracts valuable business insights such as:

- Sales trends by month, region, and category  
- Payment method distribution  
- Review sentiment & score patterns  
- Delivery delays and shipping times  
- Predictive modeling of **late deliveries**

The pipeline combines **data engineering (ETL)**, **data analytics (Athena EDA)**, and **machine learning (ML)**  
into a single automated workflow.

---

## 🧱 AWS Components Used

| Component | Purpose |
|------------|----------|
| **AWS S3** | Data Lake storage for raw → normalized → preprocessed → wrangled layers |
| **AWS Glue (ETL)** | PySpark-based normalization, cleaning, and wrangling jobs |
| **AWS Glue Crawler** | Auto-catalog S3 Parquet tables into Athena database |
| **AWS Athena** | Serverless SQL analytics engine |
| **AWS Boto3 SDK** | Automate Glue, Athena, and S3 workflows |
| **AWS Wrangler** | Pandas ↔ Athena data transfer helper |
| **Plotly + Bootstrap** | Interactive HTML dashboards |
| **scikit-learn** | Machine learning and model evaluation |

---

## 🧰 Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Language** | Python 3.12 |
| **Data Engineering** | PySpark (AWS Glue), AWS Glue Crawler |
| **Data Storage** | Amazon S3 (Raw → Normalized → Preprocessed → Wrangled Layers) |
| **Data Analytics** | AWS Athena, AWS Wrangler |
| **Visualization** | Plotly, Bootstrap, Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, joblib |
| **AWS SDKs** | boto3, botocore |
| **Utilities** | pandas, numpy |
| **Deployment** | AWS Glue ETL Jobs, Athena Workgroup, S3 Dashboards |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/salesMLCapstoneProject.git
cd salesMLCapstoneProject
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
```

Activate it:
- **Windows (PowerShell)**  
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**  
  ```bash
  source venv/bin/activate
  ```

### 3️⃣ Upgrade pip and Tooling
```bash
pip install --upgrade pip setuptools wheel
```

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
> 💡 *If `requirements_resolved.txt` is not available, use `requirements.txt` instead.*

### 5️⃣ Configure AWS Credentials

Ensure your environment has valid AWS credentials with access to:
- The S3 bucket defined in `config/config.json`
- AWS Glue and Athena services
- IAM role `AWSGlueServiceRole-ETL`

Verify access:
```bash
aws sts get-caller-identity
```

### 6️⃣ Run the Full Pipeline
```bash
python scripts/run_full_pipeline.py
```

This single command will:
- Execute all AWS Glue ETL jobs sequentially  
- Crawl and catalog data in AWS Glue  
- Run Athena EDA queries  
- Build the EDA dashboard  
- Train baseline and enhanced ML models  
- Generate dashboards under the `output/` directory

---

## 📂 Project Structure

```text
salesMLCapstoneProject/
├── config/
│   └── config.json                  # S3 paths, Glue role, bucket, crawler, database
├── scripts/
│   ├── glue_normalize_denorm.py     # Step 1 - Normalize
│   ├── glue_understand_tables.py    # Step 2 - Inspect tables
│   ├── glue_clean_tables.py         # Step 3 - Clean data
│   ├── glue_wrangle_data.py         # Step 4 - Merge & wrangle
│   ├── athena_eda_runner.py         # Step 5 - Athena EDA queries
│   ├── build_html_dashboard.py      # Step 6 - EDA dashboard
│   ├── run_ml_model.py              # Step 7 - Base ML (LogReg, RF)
│   ├── run_ml_model_enhanced.py     # Step 8 - Enhanced ML (GB, SVC, metrics)
│           
├── data/                            # Cached wrangled dataset
├── output/
│   ├── olist_eda_dashboard.html     # EDA dashboard
│   ├── olist_ml_dashboard.html      # Enhanced ML dashboard
│   └── models/                      # Trained models + scaler
├── requirements.txt
├── run_full_pipeline.py             # Master orchestrator
└── README.md

```

---

## 🧠 Machine Learning Overview

### Base Models (`run_ml_model.py`)
- Logistic Regression  
- Random Forest  
- Metrics: AUC, Accuracy  
- Visuals: ROC Curves, Feature Importance, Confusion Matrix  

### Enhanced Models (`run_ml_model_enhanced.py`)
- Gradient Boosting & SVC  
- Metrics: Precision, Recall, F1-Score, Training Time  
- Feature Correlation Heatmap  
- Class Balance Chart  
- Model Comparison Table  
- Auto-saves `.pkl` models in `/output/models`

---

## 📊 Dashboard Outputs

| File | Description |
|-------|--------------|
| `output/olist_eda_dashboard.html` | Interactive EDA dashboard |
| `output/olist_ml_dashboard.html` | ML-focused dashboard |
| `output/models/*.pkl` | Saved trained models and scaler |

---

## 🧑‍💻 Authors & Acknowledgments

**Developed by:** Group G16 – Ankit, Sarthak, Kaushal, Saransh <br>
**Mentors:** *[Add Faculty / Industry Mentor Names]*  
**Dataset:** [Olist Brazilian E-Commerce Dataset (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

✨ *From Raw Data to Insightful Dashboards and Predictive Models – all in one automated AWS pipeline.*
