# Databricks notebook source
dbutils.widgets.text("layer", "bronze")
layer = dbutils.widgets.get("layer")

print("Running layer:", layer)

# COMMAND ----------

from pyspark.sql import functions as F

if layer == "bronze":
    
    events = spark.read.table("workspace.default.mobile_sales")
    
    events.write.format("delta") \
        .mode("overwrite") \
        .saveAsTable("workspace.default.mobile_sales_delta")
    
    print("Bronze layer refreshed successfully")


elif layer == "silver":
    
    events = spark.read.table("workspace.default.mobile_sales_delta")
    
    features_df = events.groupBy("Customer_Name").agg(
        F.count("*").alias("total_transactions"),
        F.sum("Quantity").alias("total_quantity"),
        F.sum("Price").alias("total_spent"),
        F.avg("Price").alias("avg_price"),
        F.countDistinct("Mobile_Model").alias("unique_models_purchased")
    )
    
    features_df.write.format("delta") \
        .mode("overwrite") \
        .saveAsTable("workspace.default.user_features_silver")
    
    print("Silver layer refreshed successfully")

else:
    print("Invalid layer selected")