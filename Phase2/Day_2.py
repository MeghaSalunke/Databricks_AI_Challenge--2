# Databricks notebook source
# DBTITLE 1,Cell 1
events = spark.read.table("workspace.default.mobile_sales_delta")

# COMMAND ----------

# DBTITLE 1,Untitled
from pyspark.sql import functions as F

features_df = events.groupBy("CustomerGender").agg(
    F.count("*").alias("total_transactions"),
    F.sum("UnitsSold").alias("total_quantity"),
    F.sum("Price").alias("total_spent"),
    F.avg("Price").alias("avg_price"),
    F.countDistinct("MobileModel").alias("unique_models_purchased")
)

# COMMAND ----------

# DBTITLE 1,Cell 3
print("Total rows:", features_df.count())

print("Unique customers:", 
      features_df.select("CustomerGender").distinct().count())

# COMMAND ----------

features_df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in features_df.columns
]).show()

# COMMAND ----------

features_df.describe().show()

# COMMAND ----------

features_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.user_features_silver")