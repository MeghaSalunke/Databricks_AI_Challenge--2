# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Lakehouse Architecture
# MAGIC
# MAGIC This project follows the Databricks Lakehouse architecture using the Medallion pattern:
# MAGIC
# MAGIC Bronze → Silver → Gold → ML → Inference → Recommendation
# MAGIC
# MAGIC ## Data Flow
# MAGIC
# MAGIC Raw Sales Data → Bronze Delta Table  
# MAGIC → Feature Engineering → Silver Table  
# MAGIC → ML Training Dataset → Model Training  
# MAGIC → MLflow Experiment Tracking  
# MAGIC → Batch Inference → Gold Predictions Table  
# MAGIC → Recommendation System

# COMMAND ----------

spark.sql("SHOW TABLES IN workspace.default").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize Bronze Layer

# COMMAND ----------

bronze_df = spark.read.table("workspace.default.mobile_sales_delta")

print("Bronze Layer Record Count:", bronze_df.count())

display(bronze_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize Silver Layer

# COMMAND ----------

silver_df = spark.read.table("workspace.default.user_features_silver")

print("Silver Layer Record Count:", silver_df.count())

display(silver_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize Gold Layer

# COMMAND ----------

# DBTITLE 1,Gold Table Demo
gold_df = spark.read.table("workspace.default.gold_product_revenue")

print("Gold Layer Record Count:", gold_df.count())

display(gold_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Architecture Diagram
# MAGIC
# MAGIC                 +--------------------+
# MAGIC                 |   Raw CSV Dataset  |
# MAGIC                 +----------+---------+
# MAGIC                            |
# MAGIC                            v
# MAGIC                 +--------------------+
# MAGIC                 |   Bronze Layer     |
# MAGIC                 | mobile_sales_delta |
# MAGIC                 +----------+---------+
# MAGIC                            |
# MAGIC                            v
# MAGIC                 +--------------------+
# MAGIC                 |   Silver Layer     |
# MAGIC                 | user_features_silver|
# MAGIC                 +----------+---------+
# MAGIC                            |
# MAGIC                            v
# MAGIC                 +--------------------+
# MAGIC                 |  ML Training       |
# MAGIC                 | Logistic / RF      |
# MAGIC                 +----------+---------+
# MAGIC                            |
# MAGIC                            v
# MAGIC                 +--------------------+
# MAGIC                 |  MLflow Tracking   |
# MAGIC                 +----------+---------+
# MAGIC                            |
# MAGIC                            v
# MAGIC                 +--------------------+
# MAGIC                 | Batch Inference    |
# MAGIC                 +----------+---------+
# MAGIC                            |
# MAGIC                            v
# MAGIC                 +--------------------+
# MAGIC                 |   Gold Layer       |
# MAGIC                 | Predictions Table  |
# MAGIC                 +----------+---------+
# MAGIC                            |
# MAGIC                            v
# MAGIC                 +--------------------+
# MAGIC                 | Recommendation ALS |
# MAGIC                 +--------------------+

# COMMAND ----------

# MAGIC %md
# MAGIC # Pipeline Flow
# MAGIC
# MAGIC 1. Raw sales data is ingested into the Bronze Delta table.
# MAGIC 2. Data is cleaned and aggregated to create user-level features in the Silver layer.
# MAGIC 3. Machine learning models are trained using Spark ML.
# MAGIC 4. Experiments are tracked using MLflow.
# MAGIC 5. The best model is used for batch inference to generate predictions.
# MAGIC 6. Predictions are stored in the Gold Delta table.
# MAGIC 7. ALS recommendation system generates personalized product suggestions.

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Retraining Strategy
# MAGIC
# MAGIC To maintain model accuracy, retraining should occur periodically.
# MAGIC
# MAGIC ## Retraining Plan
# MAGIC
# MAGIC Daily
# MAGIC - New data ingestion into Bronze table
# MAGIC
# MAGIC Weekly
# MAGIC - Feature table refresh
# MAGIC - Model retraining pipeline execution
# MAGIC
# MAGIC Monthly
# MAGIC - Hyperparameter tuning
# MAGIC - Model evaluation
# MAGIC
# MAGIC Trigger-based retraining
# MAGIC - Significant drop in model AUC
# MAGIC - Large increase in new users/products
# MAGIC
# MAGIC ## Automation
# MAGIC
# MAGIC Retraining pipeline can be scheduled using Databricks Jobs.