# Databricks notebook source
# DBTITLE 1,Imports
import os

# COMMAND ----------

# DBTITLE 1,Query dataset
df = spark.read.json(f"file:{os.getcwd()}/simple-ml") # "file:" prefix and absolute file path are required for PySpark
df.orderBy('value2').display()

# COMMAND ----------

# DBTITLE 1,Feature Engineering with Transformers
# When we use MLlib, all inputs to ML algorithms (with severla exceptions discussed in later chapters) in Spark must consist of type Double (for labels) and Vector[Double] (for features)

# RFormula supports a limited subset of the R operators that in practice work quite well for simple models and manipulations.
from pyspark.ml.feature import RFormula
supervised = RFormula(formula = 'lab ~ . + color: value1 + color: value2')

# COMMAND ----------

# DBTITLE 1,Preparing the Dataframe
fittedRF = supervised.fit(df)
preparedDF = fittedRF.transform(df)
preparedDF.display()

# COMMAND ----------

# DBTITLE 1,Train Test Split
train, test = preparedDF.randomSplit([.7,.3])

# COMMAND ----------

# DBTITLE 1,Call the Logistic Regressor model
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol = 'label', featuresCol = 'features')
print(lr.explainParams())

# COMMAND ----------

# DBTITLE 1,Train the model it is used .fit function
fittedLR = lr.fit(train)

# COMMAND ----------

# DBTITLE 1,Check the results
fittedLR.transform(train).display()

# COMMAND ----------

# DBTITLE 1,Train Test Split for Pipeline workflow
train_pipeline, test_pipeline = df.randomSplit([.7,.3])

# COMMAND ----------

# DBTITLE 1,Recreating Dataframe preparing and call ML Model
rForm = RFormula()
lr = LogisticRegression().setLabelCol('label').setFeaturesCol('features')

# COMMAND ----------

# DBTITLE 1,Creating one stage (Pipeline)
# Now every transformation and stages will be added into one list
from pyspark.ml import Pipeline
stages = [rForm, lr]
pipeline = Pipeline().setStages(stages)

# COMMAND ----------

# DBTITLE 1,Create test for different combinations of hyperparameters in models
# 2 Formulas, 3 Elastic Net, 2 Regularization - Total 12 possibilities
from pyspark.ml.tuning import ParamGridBuilder
params = ParamGridBuilder()\
  .addGrid(rForm.formula, [
  "lab ~ . + color: value1",
  "lab ~ . + color: value1 + color: value2"])\
  .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
  .addGrid(lr.regParam, [0.1, 2.0])\
  .build()

# COMMAND ----------

# DBTITLE 1,Evaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()\
  .setMetricName('areaUnderROC')\
  .setRawPredictionCol('prediction')\
  .setLabelCol('label')

# COMMAND ----------

# DBTITLE 1,Creation of special subset for Hyperparameters optimization to avoid overfitting
from pyspark.ml.tuning import TrainValidationSplit
tvs = TrainValidationSplit()\
  .setTrainRatio(0.75)\
  .setEstimatorParamMaps(params)\
  .setEstimator(pipeline)\
  .setEvaluator(evaluator)

# COMMAND ----------

# DBTITLE 1,Train call
# the output will be model
tvsFitted = tvs.fit(train_pipeline)

# COMMAND ----------

# DBTITLE 1,Evaluate how it performs in test dataset
evaluator.evaluate(tvsFitted.transform(test_pipeline))
