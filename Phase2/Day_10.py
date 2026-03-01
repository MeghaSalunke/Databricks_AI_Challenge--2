# Databricks notebook source
events = spark.read.table("workspace.default.mobile_sales_delta")

# COMMAND ----------

# MAGIC %md
# MAGIC Run a Heavy Query

# COMMAND ----------

heavy_query = spark.sql("""
SELECT CustomerGender,
       SUM(Price) as total_spent,
       COUNT(*) as total_transactions
FROM workspace.default.mobile_sales_delta
WHERE Price > 50000
GROUP BY CustomerGender
""")

heavy_query.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Analyze Explain Plan

# COMMAND ----------

spark.sql("""
SELECT CustomerGender,
       SUM(Price) as total_spent
FROM workspace.default.mobile_sales_delta
WHERE Price > 50000
GROUP BY CustomerGender
""").explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC Enable Caching

# COMMAND ----------

heavy_query_cached = spark.sql("""
SELECT CustomerGender,
       SUM(Price) as total_spent,
       COUNT(*) as total_transactions
FROM workspace.default.mobile_sales_delta
WHERE Price > 50000
GROUP BY CustomerGender
""")

heavy_query_cached.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Compare Execution Time

# COMMAND ----------

import time

start = time.time()

spark.sql("""
SELECT CustomerGender,
       SUM(Price) as total_spent
FROM workspace.default.mobile_sales_delta
WHERE Price > 50000
GROUP BY CustomerGender
""").show()

end = time.time()

print("Execution time:", round(end - start, 3), "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC Demonstrate Partition Pruning

# COMMAND ----------

events.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("CustomerGender") \
    .saveAsTable("workspace.default.mobile_sales_partitioned")

# COMMAND ----------

spark.sql("""
SELECT *
FROM workspace.default.mobile_sales_partitioned
WHERE CustomerGender = 'Male'
""").explain(True)