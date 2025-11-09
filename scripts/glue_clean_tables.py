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
        print(f"📥 Loaded {name} from {path} ({df.count()} rows)")
        return df
    except AnalysisException as e:
        print(f"⚠️ Missing table {name}: {e.desc}")
        return None
    except Exception as e:
        print(f"⚠️ Could not read {name}: {e}")
        return None

# Helper Functions to write data to s3 bucket.
def safe_write(df, name):
    if df is None:
        print(f"⚠️ Skipping write for {name}")
        return
    dest = f"{PRE}/{name}"
    df.write.mode("overwrite").parquet(dest)
    print(f"✅ Saved {name} → {dest} ({df.count()} rows)")

# ============================================================
# 1️⃣ ORDERS (verified timestamp fix)
# ============================================================
orders = safe_read("orders")
if orders is not None:
    print("🧹 Cleaning orders table...")

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for c in date_cols:
        if c in orders.columns:
            dtype = [d for d in orders.dtypes if d[0] == c][0][1]

            # Case 1: epoch milliseconds (long)
            if dtype in ("bigint", "long", "int"):
                orders = orders.withColumn(
                    c, F.from_unixtime((F.col(c) / 1000).cast("long")).cast("timestamp")
                )
                print(f"✅ Converted {c} from epoch → timestamp")

            # Case 2: string (M/d/yyyy H:mm)
            elif dtype == "string":
                orders = orders.withColumn(
                    c, F.to_timestamp(F.col(c), "M/d/yyyy H:mm")
                )
                print(f"✅ Parsed {c} from string → timestamp")

            # Case 3: already timestamp
            elif dtype == "timestamp":
                print(f"⏱️ {c} already timestamp")

            else:
                # fallback generic cast
                orders = orders.withColumn(
                    c, F.from_unixtime(F.col(c).cast("long")).cast("timestamp")
                )
                print(f"⚙️ Fallback cast for {c} ({dtype}) → timestamp")

    # Remove impossible timeline rows
    cond_invalid = (
        (F.col("order_approved_at") < F.col("order_purchase_timestamp")) |
        (F.col("order_delivered_carrier_date") < F.col("order_approved_at")) |
        (F.col("order_delivered_customer_date") < F.col("order_delivered_carrier_date")) |
        (F.col("order_estimated_delivery_date") < F.col("order_purchase_timestamp"))
    )
    invalid_orders = orders.filter(cond_invalid)
    print(f"⚠️ Invalid orders dropped: {invalid_orders.count()}")
    orders = orders.subtract(invalid_orders).dropDuplicates(["order_id"])

    # Force all timestamps to real Spark TIMESTAMP type
    exprs = [
        f"CAST({c} AS TIMESTAMP) AS {c}" if c in date_cols else c
        for c in orders.columns
    ]
    orders_final = orders.selectExpr(*exprs)

    # ✅ Parquet writer settings to preserve timestamp type
    cleanSpark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
    cleanSpark.conf.set("spark.sql.session.timeZone", "UTC")

    dest = f"{PRE}/orders"
    orders_final.write.mode("overwrite") \
        .option("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS") \
        .option("compression", "snappy") \
        .parquet(dest)

    print(f"✅ Orders written with correct TIMESTAMP columns → {dest}")

# ============================================================
# 2️⃣ ORDER ITEMS
# ============================================================
items = safe_read("order_items")
if items is not None:
    items = items.filter((F.col("price") > 0) & (F.col("freight_value") >= 0))
    if "shipping_limit_date" in items.columns:
        items = items.withColumnRenamed("shipping_limit_date", "shipping_deadline")
        items = items.withColumn("shipping_deadline", F.to_timestamp("shipping_deadline"))
    safe_write(items, "order_items")

# ============================================================
# 3️⃣ ORDER PAYMENTS
# ============================================================
pays = safe_read("order_payments")
if pays is not None:
    pays = pays.withColumn(
        "payment_type",
        F.when(F.col("payment_type") == "boleto", "bank_slip")
         .otherwise(F.col("payment_type"))
    )
    pays = pays.filter(F.col("payment_type") != "not_defined")
    safe_write(pays, "order_payments")

# ============================================================
# 4️⃣ ORDER REVIEWS
# ============================================================
reviews = safe_read("order_reviews")
if reviews is not None:
    for c in ["review_creation_date", "review_answer_timestamp"]:
        if c in reviews.columns:
            reviews = reviews.withColumn(c, F.to_timestamp(F.col(c)))
    reviews = reviews.filter(F.col("review_id").isNotNull())
    before = reviews.count()
    reviews = reviews.dropDuplicates(["review_id"])
    after = reviews.count()
    print(f"🧹 Cleaned reviews: removed {before - after} duplicates")
    safe_write(reviews, "order_reviews")

# ============================================================
# 5️⃣ PRODUCTS
# ============================================================
products = safe_read("products")
if products is not None:
    # fix typos + ensure length columns exist
    rename_map = {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
    for old, new in rename_map.items():
        if old in products.columns:
            products = products.withColumnRenamed(old, new)
        elif new not in products.columns:
            products = products.withColumn(new, F.lit(None).cast("double"))

    if "product_category_name" in products.columns:
        products = products.filter(F.col("product_category_name").isNotNull())

    # fill numeric fields with median values
    num_cols = [
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm"
    ]
    for c in num_cols:
        if c in products.columns:
            products = products.withColumn(c, F.col(c).cast(DoubleType()))
            non_null = products.filter(F.col(c).isNotNull())
            if non_null.count() > 0:
                med = non_null.approxQuantile(c, [0.5], 0.01)[0]
                products = products.na.fill({c: float(med)})
            products = products.filter(F.col(c) > 0)

    safe_write(products, "products")

# ============================================================
# 6️⃣ PRODUCT CATEGORY TRANSLATION
# ============================================================
prod_cat = safe_read("product_category_name_translation")
if prod_cat is not None:
    prod_cat = prod_cat.dropDuplicates(["product_category_name"])
    add_rows = [
        ("portateis_cozinha_e_preparadores_de_alimentos", "portable_kitchen_and_food_preparers"),
        ("pc_gamer", "pc_gamer"),
    ]
    add_df = cleanSpark.createDataFrame(add_rows, ["product_category_name", "product_category_name_english"])
    prod_cat = prod_cat.unionByName(add_df, allowMissingColumns=True).dropDuplicates(["product_category_name"])
    safe_write(prod_cat, "product_category_name_translation")

# ============================================================
# 7️⃣ SELLERS
# ============================================================
sellers = safe_read("sellers")
if sellers is not None:
    @F.udf(StringType())
    def strip_accents(text):
        if text is None:
            return None
        return ''.join(
            c for c in unicodedata.normalize('NFKD', str(text))
            if not unicodedata.combining(c)
        )

    sellers = sellers.dropDuplicates(["seller_id"]).filter(F.col("seller_id").isNotNull())
    if "seller_city" in sellers.columns:
        sellers = sellers.withColumn("seller_city", F.lower(F.trim(F.col("seller_city"))))
        sellers = sellers.withColumn("seller_city", strip_accents("seller_city"))

        replacements = {
            "sao miguel": "sao miguel d'oeste",
            "sao pau": "sao paulo",
        }
        for wrong, correct in replacements.items():
            sellers = sellers.withColumn(
                "seller_city",
                F.when(F.col("seller_city").contains(wrong), correct)
                 .otherwise(F.col("seller_city"))
            )
    safe_write(sellers, "sellers")

# ============================================================
# 8️⃣ CUSTOMERS + GEOLOCATION
# ============================================================
cust = safe_read("customers")
if cust is not None:
    cust = cust.dropDuplicates(["customer_id"]).filter(F.col("customer_id").isNotNull())
    safe_write(cust, "customers")

geo = safe_read("geolocation")
if geo is not None:
    geo = geo.dropDuplicates(["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"])
    geo = geo.filter(
        (F.col("geolocation_lat").between(-90, 90)) &
        (F.col("geolocation_lng").between(-180, 180))
    )
    safe_write(geo, "geolocation")

print("\n🎉 Cleaning job completed successfully (Final Version).")
job.commit()
