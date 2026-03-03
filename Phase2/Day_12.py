# Databricks notebook source
# MAGIC %md
# MAGIC Analyze Job Runtime

# COMMAND ----------

import time

start = time.time()

spark.sql("""
SELECT CustomerGender,
       SUM(Price) as total_revenue
FROM workspace.default.mobile_sales_delta
GROUP BY CustomerGender
""").show()

end = time.time()

print("Execution Time (No Cache):", round(end - start, 3), "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC Enable Caching

# COMMAND ----------

# DBTITLE 1,Cell 2
spark.table("workspace.default.mobile_sales_delta").count()

# COMMAND ----------

start = time.time()

spark.sql("""
SELECT CustomerGender,
       SUM(Price) as total_revenue
FROM workspace.default.mobile_sales_delta
GROUP BY CustomerGender
""").show()

end = time.time()

print("Execution Time (With Cache):", round(end - start, 3), "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC Reduce Unnecessary Actions

# COMMAND ----------

df = spark.table("workspace.default.mobile_sales_delta")

df.count()
df.show()
df.describe().show()

# COMMAND ----------

# DBTITLE 1,Cell 5
df = spark.table("workspace.default.mobile_sales_delta")

df.count()  # materialize once
df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Demonstrate Repartition vs Coalesce

# COMMAND ----------

# DBTITLE 1,Cell 6
optimized_df = df.coalesce(2)
print("Reduced partitions (approximate):", optimized_df.count())

# COMMAND ----------

spark.table("workspace.default.mobile_sales_delta") \
    .write.format("delta") \
    .mode("overwrite") \
    .partitionBy("CustomerGender") \
    .saveAsTable("workspace.default.mobile_sales_optimized")

# COMMAND ----------

# MAGIC %md
# MAGIC Data Partitioning Strategy

# COMMAND ----------

spark.sql("""
SELECT *
FROM workspace.default.mobile_sales_optimized
WHERE CustomerGender = 'Male'
""").explain(True)