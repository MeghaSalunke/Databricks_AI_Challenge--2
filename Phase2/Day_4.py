# Databricks notebook source
# MAGIC %md
# MAGIC Read Bronze Table Schema

# COMMAND ----------

events = spark.read.table("workspace.default.mobile_sales_delta")

# COMMAND ----------

# MAGIC %md
# MAGIC Create Streaming DataFrame From Bronze Table

# COMMAND ----------

stream_df = spark.readStream \
    .table("workspace.default.mobile_sales_delta")

# COMMAND ----------

# MAGIC %md
# MAGIC Write Streaming Output to New Delta Table

# COMMAND ----------

# DBTITLE 1,Streaming write with Unity Catalog checkpoint
# Ensure stream_df is defined before running
try:
    query = stream_df.writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", "/Volumes/workspace/default/stream_volume/checkpoint") \
        .trigger(availableNow=True) \
        .toTable("workspace.default.mobile_sales_stream_bronze")
except NameError:
    print("stream_df is not defined. Please run Cell 2 first.")

# COMMAND ----------

query.status

# COMMAND ----------

spark.sql("""
SELECT * 
FROM workspace.default.mobile_sales_stream_bronze
""").display()

# COMMAND ----------

query.stop()