# Databricks notebook source
# MAGIC %md
# MAGIC ### Initial Settings

# COMMAND ----------

# DBTITLE 1,Imports
import os
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# COMMAND ----------

# DBTITLE 1,Data Query
ratings = spark.read.text(f'file:{os.getcwd()}/sample_movielens_ratings.txt')\
  .selectExpr("split(value, '::') as col")\
  .selectExpr(
  'cast(col[0] as int) as userId',
  'cast(col[1] as int) as movieId',
  'cast(col[2] as float) as rating',
  'cast(col[3] as long) as timestamp')
ratings.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recommendation Example

# COMMAND ----------

# DBTITLE 1,Data transformation
training, test = ratings.randomSplit([0.8, 0.2])
als = ALS()\
  .setMaxIter(5)\
  .setRegParam(0.01)\
  .setUserCol('userId')\
  .setItemCol('movieId')\
  .setRatingCol('rating')
#print(als.explainParams())
alsModel = als.fit(training)
predictions = alsModel.transform(test)

# COMMAND ----------

# DBTITLE 1,Predict Recommendation
alsModel.recommendForAllUsers(10)\
  .selectExpr('userId', 'explode(recommendations)').display()
alsModel.recommendForAllItems(10)\
  .selectExpr('movieId', 'explode(recommendations)').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluators for Recommendation

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()\
  .setMetricName('rmse')\
  .setLabelCol('rating')\
  .setPredictionCol('prediction')
rmse = evaluator.evaluate(predictions)
print('Root-mean-square error = %f' % rmse)
