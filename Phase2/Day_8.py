# Databricks notebook source
import mlflow
import mlflow.spark
from pyspark.sql import functions as F

# COMMAND ----------

features_df = spark.read.table("workspace.default.user_features_silver")

from pyspark.ml.feature import VectorAssembler

feature_cols = [
    "total_transactions",
    "total_quantity",
    "total_spent",
    "avg_price",
    "unique_models_purchased"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

data = assembler.transform(features_df)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="purchased",
    numTrees=100
)

# Recreate label like Day 5
events = spark.read.table("workspace.default.mobile_sales_delta")

from pyspark.sql import functions as F

label_df = events.groupBy("CustomerGender") \
    .agg(F.sum("Price").alias("customer_total_spent")) \
    .withColumn(
        "purchased",
        F.when(F.col("customer_total_spent") > 50000, 1).otherwise(0)
    ) \
    .select("CustomerGender", "purchased")

training_data = data.join(label_df, "CustomerGender")

model = rf.fit(training_data)

# COMMAND ----------

predictions = model.transform(training_data)