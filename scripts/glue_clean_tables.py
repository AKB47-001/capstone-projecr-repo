import sys, unicodedata
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.utils import AnalysisException

# Adding support for handling runtime options
args = getResolvedOptions(sys.argv, ["JOB_NAME", "RAW_S3", "PREPROCESSED_S3"])
RAW = args["RAW_S3"].rstrip("/")
PRE = args["PREPROCESSED_S3"].rstrip("/")

# Performing the configuration part
glueCleanSparkContextObject = SparkContext()
cleanGlueContext = GlueContext(glueCleanSparkContextObject)
cleanSpark = cleanGlueContext.spark_session
job = Job(cleanGlueContext)
job.init(args["JOB_NAME"], args)
cleanSpark.sparkContext.setLogLevel("WARN")

# Helper Functions to read the normalised data from s3 bucket.
# The Path on the S3 is s3://g16-capstone-iiitj/raw/normalized
def safe_read(name):
    path = f"{RAW}/{name}"
    try:
        df = cleanSpark.read.parquet(path)
        print(f"Loaded {name} from {path} ({df.count()} rows)")
        return df
    except AnalysisException as e:
        print(f"Missing table {name}: {e.desc}")
        return None
    except Exception as e:
        print(f"Could not read {name}: {e}")
        return None

# Helper Functions to write data to s3 bucket.
def safe_write(df, name):
    if df is None:
        print(f"Skipping write for {name}")
        return
    dest = f"{PRE}/{name}"
    df.write.mode("overwrite").parquet(dest)
    print(f"Saved {name} → {dest} ({df.count()} rows)")


# ------------------------------------------------
# Cleaning the Orders Table Data - Preprocessing
# ------------------------------------------------
orders_df  = safe_read("orders")
if orders_df is not None:
    print("Cleaning the ORDERS table.....")
    # Array of the columns with timestamps values
    timestamp_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    # Converting timestamps to valid Spark TimestampType
    for col_name in timestamp_columns:
        if col_name in orders_df.columns:
            dtype = [d for d in orders_df.dtypes if d[0] == col_name][0][1]
            # Case 1: epoch milliseconds (integer type)
            if dtype in ("bigint", "long", "int"):
                orders_df = orders_df.withColumn(
                    col_name, F.from_unixtime((F.col(col_name) / 1000).cast("long")).cast("timestamp")
                )
                print(f"✅ Converted {col_name} from epoch → timestamp")
            # Case 2: string format like "M/d/yyyy H:mm"
            elif dtype == "string":
                orders_df = orders_df.withColumn(
                    col_name, F.to_timestamp(F.col(col_name), "M/d/yyyy H:mm")
                )
                print(f"✅ Parsed {col_name} from string → timestamp")
            # Case 3: already timestamp
            elif dtype == "timestamp":
                print(f"⏱️ {col_name} already timestamp")
            # Fallback: generic cast
            else:
                orders_df = orders_df.withColumn(
                    col_name, F.from_unixtime(F.col(col_name).cast("long")).cast("timestamp")
                )
                print(f"⚙️ Fallback cast for {col_name} ({dtype}) → timestamp")

    
    # Droping rows with impossible delivery timelines
    invalid_timeline = (
        (F.col("order_approved_at") < F.col("order_purchase_timestamp")) |
        (F.col("order_delivered_carrier_date") < F.col("order_approved_at")) |
        (F.col("order_delivered_customer_date") < F.col("order_delivered_carrier_date")) |
        (F.col("order_estimated_delivery_date") < F.col("order_purchase_timestamp"))
    )
    invalid_rows = orders_df.filter(invalid_timeline)
    print(f"Invalid orders dropped: {invalid_rows.count()}")
    orders_df = orders_df.subtract(invalid_rows).dropDuplicates(["order_id"])
    # Ensure all date fields are cast properly
    exprs = [f"CAST({c} AS TIMESTAMP) AS {c}" if c in timestamp_columns else c for c in orders_df.columns]
    orders_cleaned = orders_df.selectExpr(*exprs)
    # Setting timestamp writing behavior
    cleanSpark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
    cleanSpark.conf.set("spark.sql.session.timeZone", "UTC")
    # Save cleaned orders table
    destination = f"{PRE}/orders"
    orders_cleaned.write.mode("overwrite") \
        .option("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS") \
        .option("compression", "snappy") \
        .parquet(destination)
    print(f"Orders written with correct TIMESTAMP columns → {destination}")


# -----------------------------------------------------
# Cleaning the Order Items Table Data - Preprocessing
# -----------------------------------------------------
order_items_df  = safe_read("order_items")
if order_items_df is not None:
    # Removing rows with non-positive prices or negative freight values.
    order_items_df = order_items_df.filter((F.col("price") > 0) & (F.col("freight_value") >= 0))
    if "shipping_limit_date" in order_items_df.columns:
        # Renamed column to shipping_deadline for clarity.
        order_items_df  = order_items_df.withColumnRenamed("shipping_limit_date", "shipping_deadline")
        # Converted shipping_limit_date to TimeStamp
        order_items_df = order_items_df.withColumn("shipping_deadline", F.to_timestamp("shipping_deadline"))
    # Writing Data
    safe_write(order_items_df, "order_items")


# -----------------------------------------------------
# Cleaning the Order Payments Table Data - Preprocessing
# -----------------------------------------------------
payments_df = safe_read("order_payments")
if payments_df is not None:
    # Normalized payment type names (e.g., boleto → bank slip)
    payments_df = payments_df.withColumn(
        "payment_type",
        F.when(F.col("payment_type") == "boleto", "bank_slip").otherwise(F.col("payment_type"))
    )
    # Remove undefined payment types
    payments_df = payments_df.filter(F.col("payment_type") != "not_defined")
    # Writing Data
    safe_write(payments_df, "order_payments")


# -----------------------------------------------------
# Cleaning the Order Reviews Table Data - Preprocessing
# -----------------------------------------------------
reviews_df = safe_read("order_reviews")
if reviews_df is not None:
    # Convert review timestamps
    for col_name in ["review_creation_date", "review_answer_timestamp"]:
        if col_name in reviews_df.columns:
            reviews_df = reviews_df.withColumn(col_name, F.to_timestamp(F.col(col_name)))
    # Removing missing or duplicate reviews
    reviews_df = reviews_df.filter(F.col("review_id").isNotNull())
    before = reviews_df.count()
    reviews_df = reviews_df.dropDuplicates(["review_id"])
    after = reviews_df.count()
    print(f"🧹 Cleaned reviews: removed {before - after} duplicates")
    # Writing Data
    safe_write(reviews_df, "order_reviews")

# -----------------------------------------------------
# Cleaning the Products Table Data - Preprocessing
# -----------------------------------------------------
products_df = safe_read("products")
if products_df is not None:
    # Fix column name typos product_name_lenght -> product_name_length
    rename_map = {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
    for old, new in rename_map.items():
        if old in products_df.columns:
            products_df = products_df.withColumnRenamed(old, new)
        elif new not in products_df.columns:
            products_df = products_df.withColumn(new, F.lit(None).cast("double"))

    # Droping products with missing category names
    if "product_category_name" in products_df.columns:
        products_df = products_df.filter(F.col("product_category_name").isNotNull())

    # Fill missing numeric attributes (weight, dimensions) with median values
    numeric_columns = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    for col_name in numeric_columns:
        if col_name in products_df.columns:
            products_df = products_df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
            non_null = products_df.filter(F.col(col_name).isNotNull())
            if non_null.count() > 0:
                median_value = non_null.approxQuantile(col_name, [0.5], 0.01)[0]
                products_df = products_df.na.fill({col_name: float(median_value)})
            products_df = products_df.filter(F.col(col_name) > 0)
    # Writing Data
    safe_write(products_df, "products")


# -----------------------------------------------------
# Cleaning the Product Category Table Data - Preprocessing
# -----------------------------------------------------
translations_df = safe_read("product_category_name_translation")
if translations_df is not None:
    translations_df = translations_df.dropDuplicates(["product_category_name"])
    # Add missing translation rows
    extra_rows = [
        ("portateis_cozinha_e_preparadores_de_alimentos", "portable_kitchen_and_food_preparers"),
        ("pc_gamer", "pc_gamer"),
    ]
    extra_df = cleanSpark.createDataFrame(extra_rows, ["product_category_name", "product_category_name_english"])
    translations_df = translations_df.unionByName(extra_df, allowMissingColumns=True).dropDuplicates(["product_category_name"])
    # Writing Data
    safe_write(translations_df, "product_category_name_translation")

# -----------------------------------------------------
# Cleaning the Sellers Table Data - Preprocessing
# -----------------------------------------------------
sellers_df = safe_read("sellers")
if sellers_df is not None:
    # UDF to remove accent marks from city names
    @F.udf(StringType())
    def strip_accents(text):
        if text is None:
            return None
        return ''.join(c for c in unicodedata.normalize('NFKD', str(text)) if not unicodedata.combining(c))
    # Droping duplicate and null seller IDs
    sellers_df = sellers_df.dropDuplicates(["seller_id"]).filter(F.col("seller_id").isNotNull())
    # Cleaning city names
    if "seller_city" in sellers_df.columns:
        sellers_df = sellers_df.withColumn("seller_city", F.lower(F.trim(F.col("seller_city"))))
        sellers_df = sellers_df.withColumn("seller_city", strip_accents("seller_city"))
        # Fixing common misspellings
        replacements = {"sao miguel": "sao miguel d'oeste", "sao pau": "sao paulo"}
        for wrong, correct in replacements.items():
            sellers_df = sellers_df.withColumn(
                "seller_city",
                F.when(F.col("seller_city").contains(wrong), correct).otherwise(F.col("seller_city"))
            )
    # Writing Data
    safe_write(sellers_df, "sellers")

# -----------------------------------------------------
# Cleaning the Customers and Geolocation Table Data - Preprocessing
# -----------------------------------------------------
customers_df = safe_read("customers")
if customers_df is not None:
    customers_df = customers_df.dropDuplicates(["customer_id"]).filter(F.col("customer_id").isNotNull())
    safe_write(customers_df, "customers")

geo_df = safe_read("geolocation")
if geo_df is not None:
    geo_df = geo_df.dropDuplicates(["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])
    geo_df = geo_df.filter(
        (F.col("geolocation_lat").between(-90, 90)) &
        (F.col("geolocation_lng").between(-180, 180))
    )
    safe_write(geo_df, "geolocation")

# Finally printing the complete status
print("\nCleaning job completed successfully!!!!")
job.commit()
