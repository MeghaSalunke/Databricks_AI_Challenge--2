# Databricks notebook source
spark.sql("""
DESCRIBE HISTORY workspace.default.mobile_sales_delta
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Append New Records

# COMMAND ----------

# DBTITLE 1,Append new records matching schema
from pyspark.sql import Row

new_data = [
    Row(CustomerGender="Male", Mobile_Model="iPhone 15", Price=90000, Quantity=1),
    Row(CustomerGender="Female", Mobile_Model="Samsung S23", Price=80000, Quantity=2)
]

new_df = spark.createDataFrame(new_data)

# Make DataFrame column names and types match table schema
new_df = new_df.withColumnRenamed("Mobile_Model", "MobileModel") \
             .withColumnRenamed("Quantity", "UnitsSold") \
             .withColumn("Price", new_df["Price"].cast("double"))

new_df.write.format("delta") \
    .mode("append") \
    .saveAsTable("workspace.default.mobile_sales_delta")

print("New records appended successfully")

# COMMAND ----------

#Check History Again
spark.sql("""
DESCRIBE HISTORY workspace.default.mobile_sales_delta
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Query Older Version

# COMMAND ----------

old_version_df = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .table("workspace.default.mobile_sales_delta")

old_version_df.show()

# COMMAND ----------

spark.sql("""
SELECT *
FROM workspace.default.mobile_sales_delta VERSION AS OF 0
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Compare Current vs Older Version

# COMMAND ----------

current_count = spark.table("workspace.default.mobile_sales_delta").count()
print("Current Version Row Count:", current_count)

# COMMAND ----------

old_count = old_version_df.count()
print("Old Version Row Count:", old_count)

# COMMAND ----------

print("New Records Added:", current_count - old_count)

# COMMAND ----------

# MAGIC %md
# MAGIC Rollback

# COMMAND ----------

spark.sql("""
RESTORE TABLE workspace.default.mobile_sales_delta TO VERSION AS OF 0
""")