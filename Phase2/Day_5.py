# Databricks notebook source
# MAGIC %md
# MAGIC Load Tables

# COMMAND ----------

from pyspark.sql import functions as F

events = spark.read.table("workspace.default.mobile_sales_delta")
features_df = spark.read.table("workspace.default.user_features_silver")

# COMMAND ----------

# MAGIC %md
# MAGIC Create Binary Label

# COMMAND ----------

# DBTITLE 1,Cell 2
label_df = events.groupBy("CustomerGender") \
    .agg(
        F.sum("Price").alias("customer_total_spent")
    ) \
    .withColumn(
        "purchased",
        F.when(F.col("customer_total_spent") > 50000, 1).otherwise(0)
    ) \
    .select("CustomerGender", "purchased")

# COMMAND ----------

# MAGIC %md
# MAGIC Join With Feature Table

# COMMAND ----------

# DBTITLE 1,Cell 3
training_data = features_df.join(
    label_df,
    on="CustomerGender",
    how="inner"
)

# COMMAND ----------

features_df.printSchema()
label_df.printSchema()
display(features_df.limit(5))
display(label_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Join Features + Label

# COMMAND ----------

# DBTITLE 1,Cell 5
training_data = features_df.join(
    label_df,
    on="CustomerGender",
    how="inner"
)

print("Total training rows:", training_data.count())
display(training_data.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Check Label Distribution

# COMMAND ----------

from pyspark.sql import functions as F

training_data.groupBy("purchased").count().show()

# COMMAND ----------

training_data.groupBy("purchased") \
    .count() \
    .withColumn("percentage",
                F.round(F.col("count") / training_data.count() * 100, 2)) \
    .show()

# COMMAND ----------

# MAGIC %md
# MAGIC Train/Test Split

# COMMAND ----------

train, test = training_data.randomSplit([0.8, 0.2], seed=42)

print("Train count:", train.count())
print("Test count:", test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Prepare Features for ML

# COMMAND ----------

feature_cols = [
    "total_transactions",
    "total_quantity",
    "total_spent",
    "avg_price",
    "unique_models_purchased"
]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

train = assembler.transform(train)
test = assembler.transform(test)

train.select("features", "purchased").show(5, truncate=False)