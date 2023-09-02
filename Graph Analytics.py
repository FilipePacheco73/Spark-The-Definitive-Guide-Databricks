# Databricks notebook source
# MAGIC %md
# MAGIC ### Initial Settings

# COMMAND ----------

# DBTITLE 1,Imports
import os

# COMMAND ----------

# DBTITLE 1,Data Query
bikeStations = spark.read.option('header','true')\
  .csv(f'file:{os.getcwd()}/bike-data/201508_station_data.csv')
tripData = spark.read.option('header','true')\
  .csv(f'file:{os.getcwd()}/bike-data/201508_trip_data.csv')

# COMMAND ----------

# DBTITLE 1,Defining the Graph - Vertices and Edges
stationVertices = bikeStations.withColumnRenamed('name','id').distinct()
tripEdges = tripData\
  .withColumnRenamed('Start Station','src')\
  .withColumnRenamed('End Station','dst')

# COMMAND ----------

# DBTITLE 1,Creating the Graph
from graphframes import GraphFrame
stationGraph = GraphFrame(stationVertices, tripEdges)
stationGraph.cache()

# COMMAND ----------

# DBTITLE 1,Basic Statistics about the Graph
print('Total Number of Stations: ' + str(stationGraph.vertices.count()))
print('Total Number of Trips in Graph: ' + str(stationGraph.edges.count()))
print('Total Number of Trips in Original Data: ' + str(tripData.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Querying the Graph

# COMMAND ----------

# DBTITLE 1,Counting the Edges
from pyspark.sql.functions import desc
stationGraph.edges.groupBy('src','dst').count().orderBy(desc('count')).display()

# COMMAND ----------

# DBTITLE 1,Subgraphs
townAnd7thEdges = stationGraph.edges\
  .where("src = 'Townsend at 7 th' OR dst = 'Townsend at 7 th'")
subgraph = GraphFrame(stationGraph.vertices, townAnd7thEdges)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Motif Finding

# COMMAND ----------

# Motifs are a way of expressing structural patterns in a graph. When we specify a motif, we are querying for patterns in the data instead of actual data.
motifs = stationGraph.find("(a)-[ab]->(b); (b)-[bc]->(c); (c)-[ca]->(a)")

# COMMAND ----------

from pyspark.sql.functions import expr
motifs.selectExpr("*",
                 "to_timestamp(ab.`Start Date`, 'MM/dd/yyyy HH:mm') as abStart",
                 "to_timestamp(bc.`Start Date`, 'MM/dd/yyyy HH:mm') as bcStart",
                 "to_timestamp(ca.`Start Date`, 'MM/dd/yyyy HH:mm') as caStart")\
  .where("ca.`Bike #` = bc.`Bike #`").where("ab.`Bike #` = bc.`Bike #`")\
  .where("a.id != b.id").where("b.id != c.id")\
  .where("abStart < bcStart").where("bcStart < caStart")\
  .orderBy(expr("cast(caStart as long) - cast(abStart as long)"))\
  .selectExpr("a.id","b.id","c.id","ab.`Start Date`","ca.`End Date`").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### PageRank

# COMMAND ----------

from pyspark.sql.functions import desc
ranks = stationGraph.pageRank(resetProbability=0.15, maxIter=10)
ranks.vertices.orderBy(desc('pagerank')).select('id','pagerank').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### In-Degree and Out-Degree Metrics
# MAGIC 
# MAGIC This is particularly applicable in the context of social networking because certain users may have more inbound connections (i.e., followers) than outbound connections (i.e., people they follow)

# COMMAND ----------

# DBTITLE 1,In-Degree measures the inbound trips, starts here
inDeg = stationGraph.inDegrees
inDeg.orderBy(desc('inDegree')).display()

# COMMAND ----------

# DBTITLE 1,In-Degree measures the outbound trips, ends here
outDeg = stationGraph.outDegrees
outDeg.orderBy(desc('outDegree')).display()

# COMMAND ----------

# DBTITLE 1,Degree Ratio - High Values indicate that trips end (but rarely begin), while a lower value, tell us where trips often begin (but infrequently end)
degreeRatio = inDeg.join(outDeg, 'id')\
  .selectExpr('id','double(inDegree)/double(outDegree) as degreeRatio')
degreeRatio.orderBy(desc('degreeRatio')).display()
degreeRatio.orderBy('degreeRatio').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Breadth-First Search
# MAGIC 
# MAGIC Breadth-First search will search our graph for how to connect two sets of nodes, based on the edges in te graph.

# COMMAND ----------

stationGraph.bfs(fromExpr="id = 'Townsend at 7th'",
                toExpr="id = 'Spear at Folsom'", maxPathLength=2).display()
