import boto3
import time
import json
import os

# Loading project configuration (region, bucket, database, and output paths)
config = json.load(open("config/config.json"))
AWS_REGION   = config["region"]
S3_BUCKET    = config["bucket_name"]
ATHENA_DB    = "olist_sales_db"  
ATHENA_OUTPUT_PATH = config["athena_output"]

# Creating an Athena client in the specified region
athena_client = boto3.client("athena", region_name=AWS_REGION)

# Function to run the Athena query and wait for completion
# returns the QueryExecutionId if successful (for tracking output files).
def run_query(sql_query, query_name):
    print(f"\nStarting to Run EDA query → {query_name}")    
    # Start Athena query execution
    response = athena_client.start_query_execution(
        QueryString=sql_query,
        QueryExecutionContext={"Database": ATHENA_DB},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_PATH}
    )
    query_execution_id = response["QueryExecutionId"]
    # Wait until the query completes (poll every 5 seconds)
    while True:
        time.sleep(5)
        query_status = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = query_status["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            print(f"   → {state}")
            break
    # If successful, return the QueryExecutionId (to locate CSV in S3)
    if state == "SUCCEEDED":
        print(f"Result stored at {ATHENA_OUTPUT_PATH}{query_execution_id}.csv")
        return query_execution_id
    else:
        print(f"Query failed: {query_name}")
        return None


# This function runs multiple SQL queries covering orders, payments, customers,
# products, reviews, and logistics to generate EDA summaries.
def perform_eda():
    print("\n")
    print("Performing Athena EDA on wrangled table")
    table_name = "wrangled"  

    # Defining EDA[Exploratory Data Analysis] Queries as Name, SQL Pairs
    queries = [
        
        # Orders and Sales Analysis
        ("order_status_distribution", f""" SELECT order_status, COUNT(*) AS total_orders FROM {table_name}
         GROUP BY order_status ORDER BY total_orders DESC;"""),
        ("monthly_orders_revenue", f""" SELECT date_trunc('month', order_purchase_timestamp) AS month,
          COUNT(DISTINCT order_status) AS total_orders, SUM(CAST(price AS DOUBLE)) AS total_revenue
          FROM {table_name} WHERE price <> '' GROUP BY 1 ORDER BY 1;"""),
        ("top_states_by_order_count", f""" SELECT customer_state, COUNT(DISTINCT order_status) AS total_orders
         FROM {table_name} WHERE customer_state IS NOT NULL GROUP BY customer_state ORDER BY total_orders DESC
         LIMIT 10;"""),

        # Payments Analysis
        ("payment_type_distribution", f""" SELECT payment_type, COUNT(*) AS transactions,
          SUM(CAST(payment_value AS DOUBLE)) AS total_value FROM {table_name}
         GROUP BY payment_type ORDER BY transactions DESC;"""),
        ("installments_distribution", f"""SELECT payment_installments, COUNT(*) AS num_orders FROM {table_name}           
         GROUP BY payment_installments ORDER BY payment_installments;"""),         

        # Customers Analysis
        ("orders_per_customer", f""" SELECT customer_zip_code_prefix AS customer_area,
            COUNT(DISTINCT order_status) AS num_orders FROM {table_name} GROUP BY customer_zip_code_prefix
         ORDER BY num_orders DESC;"""),
        ("customers_by_state", f""" SELECT customer_state, COUNT(DISTINCT customer_zip_code_prefix) AS num_customers
         FROM {table_name} WHERE customer_state IS NOT NULL GROUP BY customer_state ORDER BY num_customers DESC;"""),         

        # Products Analysis
        ("top_categories_by_sales", f""" SELECT product_category_name_english AS category, SUM(CAST(price AS DOUBLE)) AS total_sales,
            COUNT(DISTINCT order_status) AS num_orders FROM {table_name} WHERE product_category_name_english IS NOT NULL
            AND price <> '' GROUP BY product_category_name_english ORDER BY total_sales DESC LIMIT 15; """),
        ("product_price_distribution", f""" SELECT approx_percentile(CAST(price AS DOUBLE), 0.25) AS p25,
           approx_percentile(CAST(price AS DOUBLE), 0.50) AS median, approx_percentile(CAST(price AS DOUBLE), 0.75) AS p75,
           avg(CAST(price AS DOUBLE)) AS avg_price FROM {table_name};"""),
        ("review_score_distribution", f""" SELECT review_score, COUNT(*) AS num_reviews FROM {table_name}
         WHERE review_score IS NOT NULL GROUP BY review_score ORDER BY review_score;"""),         

        # Delivery and Logistics Analysis
        ("delivery_delay_distribution", f""" SELECT date_diff('day', order_estimated_delivery_date, order_delivered_customer_date) AS delay_days,
           COUNT(*) AS num_orders FROM {table_name} WHERE order_delivered_customer_date IS NOT NULL GROUP BY 1 ORDER BY delay_days;"""),
        ("shipping_time_distribution", f""" SELECT date_diff('day', order_purchase_timestamp, order_delivered_customer_date) AS shipping_days,
           COUNT(*) AS num_orders FROM {table_name} WHERE order_delivered_customer_date IS NOT NULL GROUP BY 1 ORDER BY shipping_days;"""),
        ("carrier_time_distribution", f""" SELECT date_diff('day', order_purchase_timestamp, order_delivered_carrier_date) AS carrier_days,
           COUNT(*) AS num_orders FROM {table_name} WHERE order_delivered_carrier_date IS NOT NULL GROUP BY 1 ORDER BY carrier_days;""")
    ]   

    # Executing all queries sequentially and collecting their QueryExecutionIds
    query_ids = {}
    for query_name, sql_query in queries:
        execution_id = run_query(sql_query, query_name)
        if execution_id:
            query_ids[query_name] = execution_id

    print("\nEDA queries completed")
    return query_ids  

# Runs the EDA workflow and saves all Athena Query IDs in JSON format
# for later retrieval by the dashboard builder script.
if __name__ == "__main__":
    query_ids = perform_eda()
    with open("athena_query_ids.json", "w") as file:
        json.dump(query_ids, file, indent=2)
