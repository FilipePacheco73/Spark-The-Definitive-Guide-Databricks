# Databricks notebook source
# MAGIC %md
# MAGIC ### Initial Settings

# COMMAND ----------

# DBTITLE 1,Imports
import os

# COMMAND ----------

# DBTITLE 1,Data Query
bInput = spark.read.format('parquet').load(f'file:{os.getcwd()}/binary-classification')\
  .selectExpr('features', 'cast(label as double) as label')
bInput.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Model: Logistic Regression

# COMMAND ----------

# DBTITLE 1,Logistic Regression Example
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression()
print(lr.explainParams()) # see all parameters
lrModel = lr.fit(bInput)

# COMMAND ----------

# DBTITLE 1,Take a look into Coefficients
print(lrModel.coefficients)
print(lrModel.intercept)

# In case of Multinominal model, lrModel.CoefficientMatrix and lrModel.interceptVector must be used.

# COMMAND ----------

# DBTITLE 1,Model Summary
summary = lrModel.summary
print(summary.areaUnderROC)
summary.roc.display()
summary.pr.display()
summary.objectiveHistory

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Model: Decision Trees

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier()
print(dt.explainParams())
dtModel = dt.fit(bInput)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Model: Random Forest and Gradient-Boosted Trees

# COMMAND ----------

# Is generated severla Random Forest each one specialized in samll part of dataset, and the effect of 'wisdom of the crowds' is applied
from pyspark.ml.classification import RandomForestClassifier
rfClassifier = RandomForestClassifier()
print(rfClassifier.explainParams())
trainedModel = rfClassifier.fit(bInput)

from pyspark.ml.classification import GBTClassifier
gbtClassifier = GBTClassifier()
print(gbtClassifier.explainParams())
trainedModel = gbtClassifier.fit(bInput)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Naive Bayes

# COMMAND ----------

# All input features must be non-negative
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()
print(nb.explainParams())
trainedModel = nb.fit(bInput.where('label != 0'))
