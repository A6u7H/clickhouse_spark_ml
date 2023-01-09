import logging
import configparser

from pyspark.ml import Pipeline
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, Normalizer
from operator import add

from consts import COLUMNS
from loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = "./config/config.ini"
model_config = configparser.ConfigParser()
model_config.read(config_path)

loader = DataLoader(model_config)

conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

uploader = DataLoader(model_config)
uploader.load()

df_datamart = spark.read.parquet(
    model_config["CLUSTERING"]["INPUT_FILE"]
).repartition(model_config.getint("CLASSIFIER", "NUM_PARTITIONS"))

(trainingData, testData) = df_datamart.randomSplit([0.7, 0.3])

cls_model = LogisticRegression(
    maxIter=10,
    family="multinomial",
    featuresCol="features",
    labelCol="cluster",
    predictionCol="class"
)

kmeans_model = KMeans(
    k=model_config.getint("CLUSTERING", "N_CLUSTERS"),
    seed=42,
    featuresCol="features",
    predictionCol="cluster"
)

assemble = VectorAssembler(inputCols=COLUMNS, outputCol="features")

pipeline = Pipeline(stages=[assemble, kmeans_model, cls_model])
model = pipeline.fit(trainingData)

classifing_summary = model.stages[2].summary

logger.info(f"\tAccuracy: {classifing_summary.accuracy}")
logger.info(f"\tPrecision: {classifing_summary.weightedPrecision}")
logger.info(f"\tRecall: {classifing_summary.weightedRecall}")
logger.info(f"\tTPR: {classifing_summary.weightedTruePositiveRate}")
logger.info(f"\tFPR: {classifing_summary.weightedFalsePositiveRate}")
logger.info(f"\tLoss history: {classifing_summary.objectiveHistory}")


predictions = model.transform(testData)

df_normed = predictions.rdd.map(lambda x: (x["class"], x["probability"])) \
    .reduceByKey(add).toDF(["class", "probability"])

normer = Normalizer(inputCol="probability", outputCol="normed", p=1)
normed = normer.transform(df_normed).rdd.map(lambda x: (x["class"], x["normed"].toArray().max()))

logger.info("Classification confidence:")
for label, confidence in normed.collect():
    logger.info(f"label: {int(label)} conf: {confidence:.4f}")
