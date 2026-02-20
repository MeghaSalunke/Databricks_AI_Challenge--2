# Databricks notebook source
# MAGIC %md
# MAGIC Load Raw CSV (Bronze Layer)

# COMMAND ----------

# DBTITLE 1,Cell 1
events = spark.read.table("workspace.default.mobile_sales")
events.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Convert to Delta Format

# COMMAND ----------

# DBTITLE 1,Cell 2
events.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.mobile_sales_delta")

# COMMAND ----------

# MAGIC %md
# MAGIC Create Managed Delta Table

# COMMAND ----------

# DBTITLE 1,Cell 3
spark.sql("""
CREATE TABLE IF NOT EXISTS workspace.default.mobile_sales_delta
USING DELTA
AS SELECT * FROM workspace.default.mobile_sales
""")

# COMMAND ----------

spark.sql("SELECT * FROM mobile_sales_delta").display()

# COMMAND ----------

# MAGIC %md
# MAGIC Simulate Small File Problem

# COMMAND ----------

# DBTITLE 1,Cell 5
for i in range(3):
    events.limit(500) \
        .write.format("delta") \
        .mode("append") \
        .saveAsTable("workspace.default.mobile_sales_delta")

# COMMAND ----------

# DBTITLE 1,Cell 6
display(spark.read.table("workspace.default.mobile_sales_delta"))

# COMMAND ----------

# MAGIC %md
# MAGIC Apply OPTIMIZE

# COMMAND ----------

spark.sql("OPTIMIZE workspace.default.mobile_sales_delta")