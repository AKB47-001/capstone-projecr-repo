import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.utils import AnalysisException

# Adding support for handling runtime options
# Arguments can be passed from the Glue console or pipeline orchestrator
args = getResolvedOptions(sys.argv, ["JOB_NAME", "PREPROCESSED_S3", "WRANGLED_S3"])
PREPROCESSED_S3_PATH = args["PREPROCESSED_S3"].rstrip("/")
WRANGLED_S3_PATH = args["WRANGLED_S3"].rstrip("/")

# Performing the configuration part
glueWrangleSparkContextObject = SparkContext()
wrangleGlueContext = GlueContext(glueWrangleSparkContextObject)
spark_session = wrangleGlueContext.spark_session
job = Job(wrangleGlueContext)
job.init(args["JOB_NAME"], args)
spark_session.sparkContext.setLogLevel("WARN")

# Reads a cleaned table from the preprocessed S3 path safely
def safe_read(table_name):    
    path = f"{PREPROCESSED_S3_PATH}/{table_name}"
    try:
        df = spark_session.read.parquet(path)
        print(f"Loaded {table_name} ({df.count()} rows)")
        return df
    except AnalysisException as e:
        print(f"Missing table {table_name}: {e.desc}")
        return None
    except Exception as e:
        print(f"Could not read {table_name}: {e}")
        return None

# Writes a DataFrame to the wrangled S3 path
def safe_write(dataframe, table_name):    
    if dataframe is None:
        print(f"Nothing to write for {table_name}")
        return
    destination_path = f"{WRANGLED_S3_PATH}/{table_name}"
    dataframe.write.mode("overwrite").parquet(destination_path)
    print(f"Saved {table_name} → {destination_path} ({dataframe.count()} rows)")

# Loading all the preprocessed tables
customers_df   = safe_read("customers")
geolocation_df = safe_read("geolocation")
orders_df      = safe_read("orders")
order_items_df = safe_read("order_items")
payments_df    = safe_read("order_payments")
reviews_df     = safe_read("order_reviews")
products_df    = safe_read("products")
translations_df= safe_read("product_category_name_translation")
sellers_df     = safe_read("sellers")


# Merging Customers + Geolocation (Temporary Enrichment)
# Adds latitude/longitude and state info to customers by matching ZIP codes.
if customers_df is not None and geolocation_df is not None:
    print("Merging customers + geolocation...")
    customers_geo_df = (
        customers_df.alias("c")
        .join(
            geolocation_df.alias("g"),
            F.col("c.customer_zip_code_prefix") == F.col("g.geolocation_zip_code_prefix"),
            "left",
        )
    )

    # Forward-fill missing geolocation values per ZIP prefix
    zip_window = (
        Window.partitionBy("customer_zip_code_prefix")
        .orderBy("customer_id")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    geo_columns = [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ]
    for col_name in geo_columns:
        if col_name in customers_geo_df.columns:
            customers_geo_df = customers_geo_df.withColumn(
                col_name, F.last(F.col(col_name), ignorenulls=True).over(zip_window)
            )

    customers_geo_df = customers_geo_df.dropDuplicates(["customer_id"])
else:
    customers_geo_df = customers_df
    print("Skipping geolocation merge (table missing)")


# Build “orders_all_df” by Merging Orders + Related Tables
# Sequentially merging orders + order_items + payments + reviews + products + category translations + sellers
orders_all_df = None
if orders_df is not None and order_items_df is not None:
    print("Building orders_all chain...")
    orders_all_df = orders_df.join(order_items_df, "order_id", "left")

    if payments_df is not None:
        orders_all_df = orders_all_df.join(payments_df, "order_id", "left")
    if reviews_df is not None:
        orders_all_df = orders_all_df.join(reviews_df, "order_id", "left")
    if products_df is not None:
        orders_all_df = orders_all_df.join(products_df, "product_id", "left")

    # Join with product category translations (English names)
    if translations_df is not None and "product_category_name" in orders_all_df.columns:
        orders_all_df = orders_all_df.join(
            translations_df.select(
                F.col("product_category_name").alias("cat_key"),
                F.col("product_category_name_english"),
            ),
            orders_all_df["product_category_name"] == F.col("cat_key"),
            "left",
        ).drop("cat_key")

    if sellers_df is not None:
        orders_all_df = orders_all_df.join(sellers_df, "seller_id", "left")

    print(f"orders_all_df complete ({orders_all_df.count()} rows)")
else:
    print("orders or order_items missing → cannot build orders_all_df")

# Combines customer and order datasets into one comprehensive dataset.
# Drops redundant columns and duplicates for analysis readiness.
commerce_df = None
if customers_geo_df is not None and orders_all_df is not None:
    print("Creating final commerce dataset...")
    commerce_df = customers_geo_df.join(orders_all_df, "customer_id", "inner")

    # Remove duplicates if any
    before_count = commerce_df.count()
    commerce_df = commerce_df.dropDuplicates()
    after_count = commerce_df.count()
    print(f"Removed {before_count - after_count} duplicate rows")

    # Drop redundant columns that are no longer needed
    columns_to_drop = [
        "customer_id",
        "customer_unique_id",
        "order_id",
        "order_item_id",
        "product_id",
        "review_id",
        "product_category_name",
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in commerce_df.columns]
    commerce_df = commerce_df.drop(*existing_columns_to_drop)

    print(f"Final commerce dataset: {commerce_df.count()} rows, {len(commerce_df.columns)} columns")
    safe_write(commerce_df, "commerce")
else:
    print("Missing dependencies → commerce dataset not created")

# Finally printing the complete status
print("\nWrangling completed – unified 'commerce' dataset created successfully.")
job.commit()
