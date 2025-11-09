import os, sys, time, json, boto3, secrets, subprocess   
from botocore.exceptions import ClientError
from scripts.athena_eda_runner import perform_eda
from scripts.build_html_dashboard import main as build_dashboard
# Allowing the imports, if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Configuration - Getting details from config.json
cfg = json.load(open("config/config.json"))
REGION       = cfg["region"]
BUCKET       = cfg["bucket_name"]
ROLE         = cfg["glue_role"]
CATALOG_DB   = cfg["catalog_database"]
CRAWLER_NAME = cfg["crawler_name"]
RAW_DENORM      = cfg["raw_denorm_path"]
NORMALIZED_S3   = cfg["normalized_layer"]
PREPROCESSED_S3 = cfg["preprocessed_layer"]
WRANGLED_S3     = cfg["wrangled_layer"]
TMP_BUCKET = f"aws-glue-scripts-{REGION}-{boto3.client('sts').get_caller_identity()['Account']}"
glue = boto3.client("glue", region_name=REGION)
s3   = boto3.client("s3", region_name=REGION)

# ============================================================
# Utility Functions
# ============================================================

def ensure_bucket_exists(bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket exists: {bucket_name}")
    except Exception:
        print(f"🪣 Creating bucket: {bucket_name}")
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": REGION}
        )

def upload_and_run_glue_job(local_script_path, default_args):
    """Uploads script to tmp S3 bucket and runs a Glue job"""
    script_name = os.path.splitext(os.path.basename(local_script_path))[0]
    job_name = f"olist_{script_name}_{secrets.token_hex(3)}"
    key = f"{job_name}.py"

    print(f"\n📤 Uploading {script_name}.py → s3://{TMP_BUCKET}/{key}")
    with open(local_script_path, "r") as f:
        s3.put_object(Bucket=TMP_BUCKET, Key=key, Body=f.read().encode("utf-8"))

    # Create Glue job
    print(f"🧩 Creating Glue job: {job_name}")
    glue.create_job(
        Name=job_name,
        Role=ROLE,
        Command={"Name": "glueetl", "PythonVersion": "3", "ScriptLocation": f"s3://{TMP_BUCKET}/{key}"},
        GlueVersion="5.0",
        DefaultArguments=default_args,
        MaxCapacity=2.0,
    )

    print(f"🚀 Starting Glue job: {job_name}")
    run_id = glue.start_job_run(JobName=job_name)["JobRunId"]

    while True:
        time.sleep(20)
        state = glue.get_job_run(JobName=job_name, RunId=run_id)["JobRun"]["JobRunState"]
        print(f"   Job {job_name} → {state}")
        if state in ["SUCCEEDED", "FAILED", "STOPPED"]:
            break

    if state != "SUCCEEDED":
        raise RuntimeError(f"❌ Glue job {job_name} failed: {state}")
    else:
        print(f"✅ {job_name} finished successfully")

    # Cleanup (optional)
    # glue.delete_job(JobName=job_name)
    return state


def ensure_crawler_and_run():
    """Ensures Glue crawler and database exist, then runs it."""
    try:
        glue.get_database(Name=CATALOG_DB)
        print(f"✅ Glue database exists: {CATALOG_DB}")
    except glue.exceptions.EntityNotFoundException:
        print(f"📚 Creating Glue database: {CATALOG_DB}")
        glue.create_database(DatabaseInput={"Name": CATALOG_DB})

    # Create crawler if not exists
    try:
        glue.get_crawler(Name=CRAWLER_NAME)
        print(f"✅ Crawler exists: {CRAWLER_NAME}")
    except glue.exceptions.EntityNotFoundException:
        print(f"🪄 Creating crawler: {CRAWLER_NAME}")
        glue.create_crawler(
            Name=CRAWLER_NAME,
            Role=ROLE,
            DatabaseName=CATALOG_DB,
            Targets={"S3Targets": [{"Path": WRANGLED_S3}]},
            SchemaChangePolicy={"UpdateBehavior": "UPDATE_IN_DATABASE", "DeleteBehavior": "LOG"}
        )

    print(f"🏃 Starting crawler: {CRAWLER_NAME}")
    glue.start_crawler(Name=CRAWLER_NAME)

    while True:
        time.sleep(20)
        state = glue.get_crawler(Name=CRAWLER_NAME)["Crawler"]["State"]
        if state == "READY":
            print("✅ Crawler finished cataloging data.")
            break


# Defining the main flow of the Pipeline Project
if __name__ == "__main__":
    print("\n**************************************************************************")
    print("🚀 Starting Sales E-Commerce - Group: G16 - ETL + EDA + MLOps Pipeline")
    print("   This project AWS Components like s3, AWS Glue, Crawler, Athena etc")
    print("****************************************************************************\n")

    ensure_bucket_exists(TMP_BUCKET)

    # -------------------------------
    # Step 1: Normalize Denormalized CSV
    # -------------------------------
    norm_args = {
        "--JOB_NAME": "normalize_denormalized_to_s3",
        "--SOURCE_PATH": RAW_DENORM,
        "--OUTPUT_S3": NORMALIZED_S3,
        "--TempDir": f"s3://{BUCKET}/temp/"
    }
    upload_and_run_glue_job("scripts/glue_normalize_denorm.py", norm_args)

    # -------------------------------
    # Step 2: Understanding Tables
    # -------------------------------
    understand_args = {
        "--JOB_NAME": "glue_understand_tables",
        "--NORMALIZED_S3": NORMALIZED_S3,
        "--TempDir": f"s3://{BUCKET}/temp/"
    }
    upload_and_run_glue_job("scripts/glue_understand_tables.py", understand_args)

    # -------------------------------
    # Step 3: Cleaning Tables
    # -------------------------------
    clean_args = {
        "--JOB_NAME": "glue_clean_tables",
        "--RAW_S3": NORMALIZED_S3,
        "--PREPROCESSED_S3": PREPROCESSED_S3,
        "--TempDir": f"s3://{BUCKET}/temp/"
    }
    upload_and_run_glue_job("scripts/glue_clean_tables.py", clean_args)

    # -------------------------------
    # Step 4: Wrangling to Commerce
    # -------------------------------
    wrangle_args = {
        "--JOB_NAME": "glue_wrangle_data",
        "--PREPROCESSED_S3": PREPROCESSED_S3,
        "--WRANGLED_S3": WRANGLED_S3,
        "--TempDir": f"s3://{BUCKET}/temp/"
    }
    upload_and_run_glue_job("scripts/glue_wrangle_data.py", wrangle_args)

    # -------------------------------
    # Step 5: Cataloging in Glue
    # -------------------------------
    ensure_crawler_and_run()

    # ---------------------------------------------
    # Step 1: Run Athena EDA Queries
    # ---------------------------------------------
    print("\n📈 Running Athena EDA...")
    ids = perform_eda()

    print("\n🧱 Generating HTML Dashboard (EDA)...")
    with open("athena_query_ids.json", "w") as f:
        json.dump(ids, f, indent=2)
    build_dashboard()
    print("✅ EDA Dashboard built successfully!")

    # ---------------------------------------------
    # Step 2: Run Machine Learning automatically
    # ---------------------------------------------
    print("\n🤖 Running ML Models: Logistic Regression vs Random Forest...")
    try:
        subprocess.run(["python", "scripts/run_ml_model.py"], check=True)
        print("✅ ML Analysis completed and appended to dashboard.")
    except subprocess.CalledProcessError as e:
        print("⚠️ ML step failed:", e)

    print("\n🎉 Pipeline completed successfully!")
    print("📊 Final Combined Dashboard → output/olist_eda_dashboard.html")

