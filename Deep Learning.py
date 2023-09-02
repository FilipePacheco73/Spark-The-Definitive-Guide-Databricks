# Databricks notebook source
# MAGIC %md
# MAGIC ### Initial Settings

# COMMAND ----------

# DBTITLE 1,Imports
import os
!pip install sparkdl
!pip install keras
!pip install tensorflow
!pip install tensorframes
!pip install resnet

# COMMAND ----------

# DBTITLE 1,Data Query
from sparkdl import readImages
img_dir = f'file:{os.getcwd()}/deep-learning-images'
image_df = readImage(img_dir)
image_df.printSchema()

# COMMAND ----------

# DBTITLE 1,NOT WORKING
import keras
import tensorflow as tf

directory = f'file:{os.getcwd()}/deep-learning-images'

tf.keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transfer Learning

# COMMAND ----------

from sparkdl import readImages
from pyspark.sql.functions import lit
tulips_df = readImages(im_dir + "/tulips").withColumn('label', lit(1))
daisy_df = readImages(im_dir + "/daisy").withColumn('label', lit(0))
tulips_train, tulips_test = tulips_df.randomSplit([0.6,0.4])
daisy_train, daisy_test = daisy_df.randomsplit([0.6,0.4])
train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)

# COMMAND ----------

# DBTITLE 1,Training Model
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
featurizer = DeepImageFeaturizer(inputCol='image',outputCol='features', modelName='InceptionV3')
lr = LogisticRegression(maxIter=1, regParam=0.05, elasticNetParam=0.3, labelCol='label')
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)

# COMMAND ----------

# DBTITLE 1,Metrics
from pyspark.ml.evaluationn import MulticlassClassificationEvaluator
tested_df = p_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
print('Test set accuracy = ' + str(evaluator.evaluate(tested_df.select('prediction','label'))))
