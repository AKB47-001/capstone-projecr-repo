import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F

# Adding support for handling runtime options
args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "SOURCE_PATH", "OUTPUT_S3"]
)
SOURCE_PATH = args["SOURCE_PATH"]
OUTPUT_S3   = args["OUTPUT_S3"].rstrip("/")

# Performing the configuration part
glueNormaliseSparkContextObject = SparkContext()
normaliseGlueContext = GlueContext(glueNormaliseSparkContextObject)
normaliseSpark = normaliseGlueContext.spark_session
job = Job(normaliseGlueContext)
job.init(args["JOB_NAME"], args)
normaliseSpark.sparkContext.setLogLevel("WARN")

# Starting to read data from the denormalised CSV file stored in S3
print(f"Reading the Denormalised CSV file from location: {SOURCE_PATH}")
df = (normaliseSpark.read.option("header", "true").option("multiLine", "true")
      .option("escape", "\"").csv(SOURCE_PATH))
print(f"Loaded All the Rows from the Denormalised CSV file. Total Row Count is: {df.count()}")
df.printSchema()
df.show(5, truncate=False)

# Table mapping with the columns stored in the denormalised dataset
TABLES = {
    "orders": [
        "order_id", "customer_id", 
        "order_status", "order_purchase_timestamp", 
        "order_approved_at", "order_delivered_carrier_date", 
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ],
    "order_items": [
        "order_id", "order_item_id", 
        "product_id", "seller_id",
        "shipping_limit_date", "price", 
        "freight_value"
    ],
    "order_payments": [
        "order_id", "payment_sequential", 
        "payment_type", "payment_installments", 
        "payment_value"
    ],
    "order_reviews": [
        "review_id", "order_id", 
        "review_score", "review_comment_title", 
        "review_comment_message", "review_creation_date", 
        "review_answer_timestamp"
    ],
    "customers": [
        "customer_id", "customer_unique_id", 
        "customer_zip_code_prefix", "customer_city", 
        "customer_state"
    ],
    "sellers": [
        "seller_id", "seller_zip_code_prefix", 
        "seller_city", "seller_state"
    ],
    "products": [
        "product_id", "product_category_name", 
        "product_name_lenght", "product_description_lenght", 
        "product_photos_qty", "product_weight_g", 
        "product_length_cm", "product_height_cm",
        "product_width_cm"
    ],
    "geolocation": [
        "geolocation_zip_code_prefix", "geolocation_lat", 
        "geolocation_lng", "geolocation_city", 
        "geolocation_state"
    ],
    "product_category_name_translation": [
        "product_category_name", 
        "product_category_name_english"
    ]
}

# Looping via each table in table Object 
for tableName, expectedColumns in TABLES.items():    
    # Check which expected columns exist
    existing = [column for column in expectedColumns if column in df.columns]
    if not existing:
        print(f"⚠️ Skipping {tableName}: none of the expected columns found!")
        continue
    # Extract those columns and drop duplicates
    nonDuplicateDF = df.select(*existing).dropDuplicates() 
    # Defining a date parsing pattern    
    DATE_PATTERN = "M/d/yyyy H:mm"   # matches "10/2/2017 10:56"
    for c in nonDuplicateDF.columns:
        if any(k in c for k in ["timestamp", "date"]):
        # cast to string first, then parse explicitly
            nonDuplicateDF = nonDuplicateDF.withColumn(c, F.to_timestamp(F.col(c).cast("string"), DATE_PATTERN))    
    # Writing normalized table to S3
    dest = f"{OUTPUT_S3}/{tableName}"
    print(f"Writing {tableName} → {dest}")
    nonDuplicateDF.write.mode("overwrite").parquet(dest)

print("We have successfully completed the Normalisation of the Data")
job.commit()
