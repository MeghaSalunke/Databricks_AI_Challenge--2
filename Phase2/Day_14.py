# Databricks notebook source
# MAGIC %md
# MAGIC # Final Production Ready System
# MAGIC
# MAGIC This notebook demonstrates the complete production pipeline:
# MAGIC
# MAGIC Data Ingestion → Feature Engineering → Model Training → 
# MAGIC MLflow Tracking → Batch Inference → Gold Predictions → Recommendations
# MAGIC
# MAGIC Key Production Considerations:
# MAGIC - Scalability
# MAGIC - Monitoring
# MAGIC - Failure Handling

# COMMAND ----------

import mlflow
import mlflow.spark

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

events = spark.read.table("workspace.default.mobile_sales_delta")

features_df = spark.read.table("workspace.default.user_features_silver")

# COMMAND ----------

label_df = events.groupBy("CustomerGender") \
    .agg(F.sum("Price").alias("customer_total_spent")) \
    .withColumn(
        "purchased",
        F.when(F.col("customer_total_spent") > 50000, 1).otherwise(0)
    ) \
    .select("CustomerGender", "purchased")

# COMMAND ----------

features_df = spark.read.table("workspace.default.user_features_silver")

print("Feature table loaded")
display(features_df.limit(10))

# COMMAND ----------

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

scoring_data = assembler.transform(features_df)

# COMMAND ----------

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="purchased",
    numTrees=100
)

model = rf.fit(train)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(
    labelCol="purchased",
    metricName="areaUnderROC"
)

predictions = model.transform(test)

auc = evaluator.evaluate(predictions)

print("Final Model AUC:", auc)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(
    labelCol="purchased",
    metricName="areaUnderROC"
)

predictions = model.transform(test)

auc = evaluator.evaluate(predictions)

print("Final Model AUC:", auc)

# COMMAND ----------

data = assembler.transform(features_df)

production_predictions = model.transform(data)

# COMMAND ----------

final_predictions = production_predictions.select(
    "CustomerGender",
    "prediction",
    "probability"
)

display(final_predictions)

# COMMAND ----------

final_predictions = production_predictions.select(
    "CustomerGender",
    "prediction",
    "probability"
)

display(final_predictions)

# COMMAND ----------

final_predictions.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.default.gold_customer_predictions")

# COMMAND ----------

# DBTITLE 1,Top buyers by prediction
from pyspark.ml.linalg import Vector
from pyspark.sql.functions import udf

# UDF to extract probability of class 1
get_prob_class_1 = udf(lambda v: float(v[1]) if v is not None else None)

# Apply UDF to extract probability

buyers_with_prob = final_predictions.withColumn("buy_probability", get_prob_class_1(F.col("probability"))) \
    .orderBy(F.desc("buy_probability"))

display(buyers_with_prob.limit(10))