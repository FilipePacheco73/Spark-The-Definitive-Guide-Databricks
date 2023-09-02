# Databricks notebook source
# MAGIC %md
# MAGIC ### Initial Settings

# COMMAND ----------

# DBTITLE 1,Imports
import os

# COMMAND ----------

# DBTITLE 1,Data Query
df = spark.read.format('parquet').load(f'file:{os.getcwd()}/regression/')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression Model: Linear Regression

# COMMAND ----------

# DBTITLE 1,Linear Regression Example
from pyspark.ml.regression import LinearRegression
lr = LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
print(lr.explainParams())
lrModel = lr.fit(df)

# COMMAND ----------

# DBTITLE 1,Training Summary
summary = lrModel.summary
summary.residuals.display()
print(summary.totalIterations)
print(summary.objectiveHistory)
print(summary.rootMeanSquaredError)
print(summary.r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression Model: Generalized Linear Regression

# COMMAND ----------

# DBTITLE 1,Logistic Regression with Parameters
from pyspark.ml.regression import GeneralizedLinearRegression
glr = GeneralizedLinearRegression()\
  .setFamily('gaussian')\
  .setLink('identity')\
  .setMaxIter(10)\
  .setRegParam(0.3)\
  .setLinkPredictionCol('linkOut')
print(glr.explainParams())
glrModel = glr.fit(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression Model: Decision Trees

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
print(dtr.explainParams())
dtrModel = dtr.fit(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression Model: Random Forests and Gradient-Boosted Trees

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
rf = RandomForestRegressor()
print(rf.explainParams())
rfModel = rf.fit(df)

gbt = GBTRegressor()
print(gbt.explainParams())
gbtModel = gbt.fit(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regression Model: Evaluatores and Automating Model Tuning

# COMMAND ----------

# DBTITLE 1,Evaluators
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

glr = GeneralizedLinearRegression().setFamily('gaussian').setLink('identity')
pipeline = Pipeline().setStages([glr])
params = ParamGridBuilder().addGrid(glr.regParam, [0, 0.5, 1]).build()
evaluator = RegressionEvaluator()\
  .setMetricName('rmse')\
  .setPredictionCol('prediction')\
  .setLabelCol('label')
cv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEvaluator(evaluator)\
  .setEstimatorParamMaps(params)\
  .setNumFolds(2) # should always be 3 or more, but this dataset is small
model = cv.fit(df)

# COMMAND ----------

# DBTITLE 1,Metrics
from pyspark.mllib.evaluation import RegressionMetrics
out = model.transform(df)\
  .select('prediction', 'label').rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = RegressionMetrics(out)
print('MSE: ' + str(metrics.meanSquaredError))
print('RMSE: ' + str(metrics.rootMeanSquaredError))
print('R-squared: ' + str(metrics.r2))
print('MAE: ' + str(metrics.meanAbsoluteError))
print('Explained variance: ' + str(metrics.explainedVariance))
