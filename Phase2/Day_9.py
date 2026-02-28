# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

# COMMAND ----------

events = spark.read.table("workspace.default.mobile_sales_delta")

# COMMAND ----------

# MAGIC %md
# MAGIC reate User-Item Interaction

# COMMAND ----------

# DBTITLE 1,Cell 3
interaction_df = events.withColumn(
    "rating",
    F.when(F.col("UnitsSold") >= 3, 3)
     .when(F.col("UnitsSold") == 2, 2)
     .otherwise(1)
).select(
    "CustomerGender",
    "MobileModel",
    "rating"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert String IDs to Numeric

# COMMAND ----------

# DBTITLE 1,Cell 4
from pyspark.ml.feature import StringIndexer

user_indexer = StringIndexer(
    inputCol="CustomerGender",
    outputCol="customer_id"
)

item_indexer = StringIndexer(
    inputCol="MobileModel",
    outputCol="product_id"
)

interaction_df = user_indexer.fit(interaction_df).transform(interaction_df)
interaction_df = item_indexer.fit(interaction_df).transform(interaction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Train ALS Model

# COMMAND ----------

als = ALS(
    userCol="user_id",
    itemCol="product_id",
    ratingCol="rating",
    coldStartStrategy="drop",
    implicitPrefs=False
)

als_model = als.fit(interaction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Generate Top Recommendations

# COMMAND ----------

# DBTITLE 1,Cell 6
from pyspark.sql.window import Window

# Generate all possible user-product pairs
user_product_pairs = interaction_df.select("user_id").distinct().crossJoin(
    interaction_df.select("product_id").distinct()
)

# Predict ratings
predictions = als_model.transform(user_product_pairs)

# Rank products for each user by predicted rating
window = Window.partitionBy("user_id").orderBy(F.col("prediction").desc())
predictions = predictions.withColumn("rank", F.row_number().over(window))

# Select top 5 recommendations per user
recommendations = predictions.filter(F.col("rank") <= 5)

display(recommendations.select("user_id", "product_id", "prediction", "rank"))

# COMMAND ----------

# MAGIC %md
# MAGIC Convert Back to Original Labels

# COMMAND ----------

user_labels = interaction_df.select("user_id", "CustomerGender").distinct()
item_labels = interaction_df.select("product_id", "Mobile_Model").distinct()

final_recommendations = recommendations.join(user_labels, "user_id")

final_recommendations.show(5, truncate=False)