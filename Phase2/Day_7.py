# Databricks notebook source
# MAGIC %md
# MAGIC Import Libraries

# COMMAND ----------

import mlflow
import mlflow.spark

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# COMMAND ----------

# MAGIC %md
# MAGIC Set MLflow Experiment

# COMMAND ----------

mlflow.set_experiment("/Users/meghasalunke2003@gmail.com/day7_model_tracking")

# COMMAND ----------

#Load Data
events = spark.read.table("workspace.default.mobile_sales_delta")
features_df = spark.read.table("workspace.default.user_features_silver")

# COMMAND ----------

# MAGIC %md
# MAGIC Create Label

# COMMAND ----------

label_df = events.groupBy("CustomerGender") \
    .agg(F.sum("Price").alias("customer_total_spent")) \
    .withColumn(
        "purchased",
        F.when(F.col("customer_total_spent") > 50000, 1).otherwise(0)
    ) \
    .select("CustomerGender", "purchased")

# COMMAND ----------

#Join
training_data = features_df.join(
    label_df,
    on="CustomerGender",
    how="inner"
)

# COMMAND ----------

train, test = training_data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

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

train = assembler.transform(train)
test = assembler.transform(test)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(
    labelCol="purchased",
    metricName="areaUnderROC"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic Regression Logging

# COMMAND ----------

with mlflow.start_run(run_name="Logistic_Regression"):

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="purchased",
        maxIter=100
    )

    lr_model = lr.fit(train)

    lr_predictions = lr_model.transform(test)

    lr_auc = evaluator.evaluate(lr_predictions)

    mlflow.log_metric("AUC", lr_auc)

    print("Logistic Regression AUC:", lr_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC Random Forest Logging

# COMMAND ----------

with mlflow.start_run(run_name="Random_Forest"):

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="purchased",
        numTrees=100
    )

    rf_model = rf.fit(train)

    rf_predictions = rf_model.transform(test)

    rf_auc = evaluator.evaluate(rf_predictions)

    mlflow.log_metric("AUC", rf_auc)

    print("Random Forest AUC:", rf_auc)
