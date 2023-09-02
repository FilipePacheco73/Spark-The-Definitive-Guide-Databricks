# Databricks notebook source
# MAGIC %md
# MAGIC ### Initial Settings

# COMMAND ----------

# DBTITLE 1,Imports
import os

# COMMAND ----------

# DBTITLE 1,Data Query
from pyspark.ml.feature import VectorAssembler
va = VectorAssembler()\
  .setInputCols(['Quantity','UnitPrice'])\
  .setOutputCol('features')

sales = va.transform(spark.read.format('csv')\
  .option('header','true')\
  .option('inferSchema','true')\
  .load(f'file:{os.getcwd()}/retail-data/by-day/*.csv')\
  .limit(50)\
  .coalesce(1)\
  .where('Description IS NOT NULL'))
sales.cache()
sales.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### K-Means

# COMMAND ----------

# DBTITLE 1,Example K-Means
from pyspark.ml.clustering import KMeans
km = KMeans().setK(5)
print(km.explainParams())
kmModel = km.fit(sales)

# COMMAND ----------

# DBTITLE 1,Metrics
summary = kmModel.summary
print(summary.clusterSizes) # number of points
#kmModel.computeCost(sales)
centers = kmModel.clusterCenters()
print('Cluster Centers: ')
for center in centers:
    print(center)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bisecting K-Means

# COMMAND ----------

# DBTITLE 1,Example Bisecting K-Means
from pyspark.ml.clustering import BisectingKMeans
bkm = BisectingKMeans().setK(5).setMaxIter(5)
bkmModel = bkm.fit(sales)

# COMMAND ----------

# DBTITLE 1,Metrics
summary = bkmModel.summary
print(summary.clusterSizes) # number of points
centers = bkmModel.clusterCenters()
print('Cluster Centers: ')
for center in centers:
  print(center)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gaussian Mixture Models - GMM

# COMMAND ----------

# DBTITLE 1,Example GMM
from pyspark.ml.clustering import GaussianMixture
gmm = GaussianMixture().setK(5)
print(gmm.explainParams())
model = gmm.fit(sales)

# COMMAND ----------

# DBTITLE 1,Metrics
summary = model.summary
print(model.weights)
model.gaussiansDF.display()
summary.cluster.display()
summary.clusterSizes
summary.probability.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latent Dirichlet Allocation - LDA

# COMMAND ----------

# DBTITLE 1,Example LDA - Data preparation (tokenizing and Vectorizing)
# Specific for Hierarchical Clustering model typically used to perform topic modelling on text documents.
from pyspark.ml.feature import Tokenizer, CountVectorizer
tkn = Tokenizer().setInputCol('Description').setOutputCol('DescOut')
tokenized = tkn.transform(sales.drop('features'))
cv = CountVectorizer()\
  .setInputCol('DescOut')\
  .setOutputCol('features')\
  .setVocabSize(500)\
  .setMinTF(0)\
  .setMinDF(0)\
  .setBinary(True)
cvFitted = cv.fit(tokenized)
prepped = cvFitted.transform(tokenized)
prepped.display()

# COMMAND ----------

# DBTITLE 1,Train - Fit
from pyspark.ml.clustering import LDA
lda = LDA().setK(10).setMaxIter(5)
print(lda.explainParams())
model = lda.fit(prepped)

# COMMAND ----------

# DBTITLE 1,Prediction
model.describeTopics(3).display()
cvFitted.vocabulary
