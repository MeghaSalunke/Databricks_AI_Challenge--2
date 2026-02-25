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

# COMMAND ----------

# MAGIC %md
# MAGIC ##DAY 6

# COMMAND ----------

# MAGIC %md
# MAGIC Import Required Libraries

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# COMMAND ----------

# MAGIC %md
# MAGIC Train Logistic Regression

# COMMAND ----------

lr = LogisticRegression(
    featuresCol="features",
    labelCol="purchased"
)

lr_model = lr.fit(train)

lr_predictions = lr_model.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate Logistic Regression (AUC)

# COMMAND ----------

evaluator = BinaryClassificationEvaluator(
    labelCol="purchased",
    metricName="areaUnderROC"
)

lr_auc = evaluator.evaluate(lr_predictions)

print("Logistic Regression AUC:", lr_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC Train Random Forest

# COMMAND ----------

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="purchased",
    numTrees=100
)

rf_model = rf.fit(train)

rf_predictions = rf_model.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate Random Forest

# COMMAND ----------

rf_auc = evaluator.evaluate(rf_predictions)

print("Random Forest AUC:", rf_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC Hyperparameter Tuning (Random Forest)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    labelCol="purchased",
    metricName="areaUnderROC"
)

best_auc = 0
best_model = None

for trees in [50, 100]:
    for depth in [5, 10]:
        
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="purchased",
            numTrees=trees,
            maxDepth=depth
        )
        
        model = rf.fit(train)
        predictions = model.transform(test)
        
        auc = evaluator.evaluate(predictions)
        
        print(f"Trees: {trees}, Depth: {depth}, AUC: {auc}")
        
        if auc > best_auc:
            best_auc = auc
            best_model = model

print("Best AUC:", best_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC Compare Models

# COMMAND ----------

# DBTITLE 1,Cell 25
print("Logistic Regression AUC:", lr_auc)
print("Random Forest AUC:", rf_auc)