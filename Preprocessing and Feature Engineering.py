# Databricks notebook source
# MAGIC %md
# MAGIC ### Initial Settings

# COMMAND ----------

# DBTITLE 1,Imports
import os

# COMMAND ----------

# DBTITLE 1,Query data
sales = spark.read.format('csv')\
  .option('header','true')\
  .option('inferSchema','true')\
  .load(f'file:{os.getcwd()}/retail-data/by-day/*.csv')\
  .coalesce(5)\
  .where('Description IS NOT NULL')
  
fakeIntDF = spark.read.parquet(f'file:{os.getcwd()}/simple-ml-integers')
simpleDF = spark.read.json(f'file:{os.getcwd()}/simple-ml')
scaleDF = spark.read.parquet(f'file:{os.getcwd()}/simple-ml-scaling')

# COMMAND ----------

# DBTITLE 1,Take a look into data
sales.cache()
sales.show()

# COMMAND ----------

# DBTITLE 1,Transformers - Convert raw data into new kind
# MAGIC %scala
# MAGIC 
# MAGIC val sales_scala = spark.read.format("csv")
# MAGIC   .option("header","true")
# MAGIC   .option("inferSchema","true")
# MAGIC   .load("file:/Workspace/Repos/PachecoFilipe@JohnDeere.com/Spark-The-Definitive-Guide/retail-data/by-day/*.csv")
# MAGIC   .coalesce(5)
# MAGIC   .where("Description IS NOT NULL")
# MAGIC   
# MAGIC import org.apache.spark.ml.feature.Tokenizer
# MAGIC val tkn = new Tokenizer().setInputCol("Description")
# MAGIC tkn.transform(sales_scala.select("Description")).show(false)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estimators for Preprocessing

# COMMAND ----------

# DBTITLE 1,StandardScale
# MAGIC %scala
# MAGIC val scaleDF_scala = spark.read.parquet("file:/Workspace/Repos/PachecoFilipe@JohnDeere.com/Spark-The-Definitive-Guide/simple-ml-scaling")
# MAGIC 
# MAGIC import org.apache.spark.ml.feature.StandardScaler
# MAGIC val ss = new StandardScaler().setInputCol("features")
# MAGIC ss.fit(scaleDF_scala).transform(scaleDF_scala).show(false)

# COMMAND ----------

# DBTITLE 1,High-Level Transformers
# Allow you to do transfomers as one, not one by one.
from pyspark.ml.feature import RFormula

supervised = RFormula(formula='lab ~. + color:value1 + color: value2')
supervised.fit(simpleDF).transform(simpleDF).display()

# COMMAND ----------

# DBTITLE 1,SQL Transformers
from pyspark.ml.feature import SQLTransformer

basicTransformation = SQLTransformer()\
  .setStatement("""
    SELECT sum(Quantity), count(*), CustomerID
    FROM __THIS__
    GROUP BY CustomerID
  """)

basicTransformation.transform(sales).display()

# COMMAND ----------

# DBTITLE 1,Vector Assembler
# VectorAssembler helps concatenate all your featuresi nto one big vector you can then pass into an estimator
from pyspark.ml.feature import VectorAssembler
va = VectorAssembler().setInputCols(['int1','int2','int3'])
va.transform(fakeIntDF).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Working with Continuous Features

# COMMAND ----------

# DBTITLE 1,Bucketing
contDF = spark.range(20).selectExpr('cast (id as double)')

# Bucketing - discretize continous data (binning)
from pyspark.ml.feature import Bucketizer
bucketBorders = [-1, 5, 10, 250, 600] # Passing limits of Bins
bucketer = Bucketizer().setSplits(bucketBorders).setInputCol('id')
bucketer.transform(contDF).display()

# Bucketing with Quantiles
from pyspark.ml.feature import QuantileDiscretizer
bucketer = QuantileDiscretizer().setNumBuckets(5).setInputCol('id')
fittedBucketer = bucketer.fit(contDF)
fittedBucketer.transform(contDF).display()

# COMMAND ----------

# DBTITLE 1,Scaling and Normalization
from pyspark.ml.feature import StandardScaler
sScaler = StandardScaler().setInputCol('features')
sScaler.fit(scaleDF).transform(scaleDF).display()

# COMMAND ----------

# DBTITLE 1,MinMaxScaler
from pyspark.ml.feature import MinMaxScaler
minMax = MinMaxScaler().setMin(5).setMax(10).setInputCol('features')
fittedminMax = minMax.fit(scaleDF)
fittedminMax.transform(scaleDF).display()

# COMMAND ----------

# DBTITLE 1,MaxAbsScaler
from pyspark.ml.feature import MaxAbsScaler
maScaler = MinMaxScaler().setInputCol('features')
fittedSCaler = maScaler.fit(scaleDF)
fittedSCaler.transform(scaleDF).display()

# COMMAND ----------

# DBTITLE 1,Elementwise Product
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors
scaleUpVec = Vectors.dense(10, 15, 20)
scalingUp = ElementwiseProduct()\
  .setScalingVec(scaleUpVec)\
  .setInputCol('features')
scalingUp.transform(scaleDF).display()

# COMMAND ----------

# DBTITLE 1,Normalizer
from pyspark.ml.feature import Normalizer
manhattanDistance = Normalizer().setP(1).setInputCol('features')
manhattanDistance.transform(scaleDF).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Working with Categorical Features

# COMMAND ----------

# DBTITLE 1,StringIndexer
from pyspark.ml.feature import StringIndexer
lblIndxr = StringIndexer().setInputCol('lab').setOutputCol('labelInd')
idxRes = lblIndxr.fit(simpleDF).transform(simpleDF)
idxRes.display()

# COMMAND ----------

# DBTITLE 1,Converting Continuous values to StringIndexer
valIndexer = StringIndexer().setInputCol('value1').setOutputCol('valueInd')
valIndexer.setHandleInvalid('skip')
valIndexer.fit(simpleDF).transform(simpleDF).display()

# COMMAND ----------

# DBTITLE 1,Converting Indexed Values Back to Text
from pyspark.ml.feature import IndexToString
labelReverse = IndexToString().setInputCol('labelInd')
labelReverse.transform(idxRes).display()

# COMMAND ----------

# DBTITLE 1,Indexing in Vectors
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors
idxIn = spark.createDataFrame([
  (Vectors.dense(1,2,3),1),
  (Vectors.dense(2,5,6),2),
  (Vectors.dense(1,8,9),3)
]).toDF('features','label')
indxr = VectorIndexer()\
  .setInputCol('features')\
  .setOutputCol('idxed')\
  .setMaxCategories(2)
indxr.fit(idxIn).transform(idxIn).display()

# COMMAND ----------

# DBTITLE 1,One-Hot Encoding
from pyspark.ml.feature import OneHotEncoder, StringIndexer
lblIndxr = StringIndexer().setInputCol('color').setOutputCol('colorInd')
colorLab = lblIndxr.fit(simpleDF).transform(simpleDF.select('color'))
ohe = OneHotEncoder().setInputCol('colorInd')
ohe.fit(colorLab).transform(colorLab).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Data Transformers

# COMMAND ----------

# DBTITLE 1,Tokenizing Text
from pyspark.ml.feature import Tokenizer
tkn = Tokenizer().setInputCol('Description').setOutputCol('DescOut')
tokenized = tkn.transform(sales.select('Description'))
tokenized.show(20, False)

# COMMAND ----------

# DBTITLE 1,Tokenizing based on RegEx
from pyspark.ml.feature import RegexTokenizer
rt = RegexTokenizer()\
  .setInputCol('Description')\
  .setOutputCol('DescOut')\
  .setPattern(' ')\
  .setGaps(True)\
  .setToLowercase(True)
rt.transform(sales.select('Description')).show(20, False)

# COMMAND ----------

# DBTITLE 1,Removing Common Words
from pyspark.ml.feature import StopWordsRemover
englishStopWords = StopWordsRemover.loadDefaultStopWords('english')
stops = StopWordsRemover()\
  .setStopWords(englishStopWords)\
  .setInputCol('DescOut')
stops.transform(tokenized).display()

# COMMAND ----------

# DBTITLE 1,Creating Word Combinations - n-grams
from pyspark.ml.feature import NGram
unigram = NGram().setInputCol('DescOut').setN(1)
bigram = NGram().setInputCol('DescOut').setN(2)
unigram.transform(tokenized.select('DescOut')).display()
bigram.transform(tokenized.select('DescOut')).display()

# COMMAND ----------

# DBTITLE 1,Converting Words into Numerical Representations - TF-IDF (Term Frequency-inverse document frequency)
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer()\
  .setInputCol('DescOut')\
  .setOutputCol('countVec')\
  .setVocabSize(500)\
  .setMinTF(1)\
  .setMinDF(2)
fittedCV = cv.fit(tokenized)
fittedCV.transform(tokenized).display()

# COMMAND ----------

# DBTITLE 1,Check the documents with 'red' word
ftIdfIn = tokenized\
  .where("array_contains(DescOut, 'red')")\
  .select('DescOut')\
  .limit(10)
ftIdfIn.display()

# COMMAND ----------

# DBTITLE 1,Word2Vec
# Transform the TF-IDF, that came from Tokenizer, to Numbers. Word2Vec is a deep learning-based tool for computing a vector representation of a set of words.
from pyspark.ml.feature import Word2Vec
# input data: Each row is a bag of words from a sentence or document
documentDF = spark.createDataFrame([
  ('Hi I heard about spark'.split(' '), ),
  ('I wish Java could use case classes'.split(' '), ),
  ('Logistics regression models are neat'.split(' '), )
], ['text'])
# Learning a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol='text', outputCol='result')
model = word2Vec.fit(documentDF)
result = model.transform(documentDF)
for row in result.collect():
  text, vector = row
  print('Text: [%s] => \nVector: %s\n' % (', '.join(text), str(vector)))

# COMMAND ----------

# DBTITLE 1,Word2Vec - Tokenized
# Transform the TF-IDF, that came from Tokenizer, to Numbers. Word2Vec is a deep learning-based tool for computing a vector representation of a set of words.
from pyspark.ml.feature import Word2Vec
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol='DescOut', outputCol='result')
model = word2Vec.fit(tokenized)
result = model.transform(tokenized)
for row in result.collect():
  Description, DescOut, vector = row
  print('Text: [%s] => \nVector: %s\n' % (', '.join(DescOut), str(vector)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Manipulation

# COMMAND ----------

# DBTITLE 1,PCA - Principal Components Analysis
from pyspark.ml.feature import PCA
pca = PCA().setInputCol('features').setK(2)
pca.fit(scaleDF).transform(scaleDF).display()

# COMMAND ----------

# DBTITLE 1,Polynomial Expansion
# To expand the number of features to create a model
from pyspark.ml.feature import PolynomialExpansion
pe = PolynomialExpansion().setInputCol('features').setDegree(2)
pe.transform(scaleDF).display()

# COMMAND ----------

# DBTITLE 1,Chi Square Selector
from pyspark.ml.feature import ChiSqSelector, Tokenizer
tkn = Tokenizer().setInputCol('Description').setOutputCol('DescOut')
tokenized = tkn\
  .transform(sales.select('Description','CustomerId'))\
  .where('CustomerId IS NOT NULL')
prechi = fittedCV.transform(tokenized)\
  .where('CustomerId IS NOT NULL')
chisq = ChiSqSelector()\
  .setFeaturesCol('countVec')\
  .setLabelCol('CustomerId')\
  .setNumTopFeatures(2)
chisq.fit(prechi).transform(prechi)\
  .drop('customerId','Description','DescOut').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Advanced Topics

# COMMAND ----------

# DBTITLE 1,Persisting Transformers
# Save
fittedPCA = pca.fit(scaleDF)
# fittedPCA.write().overwrite().save('')

# Load
# loadedPCA = PCAModel.load('/tmp/fittedPCA')
loadedPCA.transform(scaleDF).display()
