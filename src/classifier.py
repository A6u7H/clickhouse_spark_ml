import yaml
import logging
import configparser

from pyspark.ml import Pipeline
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, Normalizer
from operator import add

from loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierModel:
    def __init__(self, config):
        self.config = config
        self.loader = DataLoader(config)
        self.spark, self.spark_context = self.configurate_spark()
        with open(config["CONST"]["PATH"], 'r') as stream:
            self.project_params = yaml.safe_load(stream)

    def configurate_spark(self):
        conf = SparkConf()
        spark_context = SparkContext(conf=conf)
        spark = SparkSession(spark_context)
        return spark, spark_context

    def configure_pipeline(self):
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

        assemble = VectorAssembler(
            inputCols=self.project_params["columns"],
            outputCol="features"
        )

        self.pipeline = Pipeline(stages=[assemble, kmeans_model, cls_model])

    def prepare_data(self):
        df_datamart = self.spark.read.parquet(
            model_config["CLUSTERING"]["INPUT_FILE"]
        ).repartition(model_config.getint("CLASSIFIER", "NUM_PARTITIONS"))

        trainingData, testData = df_datamart.randomSplit([0.7, 0.3])
        return trainingData, testData

    def fit_classifier(self, training_data):
        self.model = self.pipeline.fit(training_data)
        classifing_summary = self.model.stages[2].summary

        logger.info(f"\tAccuracy: {classifing_summary.accuracy}")
        logger.info(f"\tPrecision: {classifing_summary.weightedPrecision}")
        logger.info(f"\tRecall: {classifing_summary.weightedRecall}")
        logger.info(f"\tTPR: {classifing_summary.weightedTruePositiveRate}")
        logger.info(f"\tFPR: {classifing_summary.weightedFalsePositiveRate}")
        logger.info(f"\tLoss history: {classifing_summary.objectiveHistory}")

    def predict(self, val_data):
        predictions = self.model.transform(val_data)

        df_normed = predictions.rdd.map(
            lambda x: (x["class"], x["probability"])
        ).reduceByKey(add).toDF(["class", "probability"])

        normer = Normalizer(inputCol="probability", outputCol="normed", p=1)
        normed = normer.transform(df_normed).rdd.map(
            lambda x: (x["class"], x["normed"].toArray().max())
        )

        logger.info("Classification confidence:")
        for label, confidence in normed.collect():
            logger.info(f"label: {int(label)} conf: {confidence:.4f} ")


if __name__ == "__main__":
    config_path = "./config/config.ini"
    model_config = configparser.ConfigParser()
    model_config.read(config_path)

    loader = DataLoader(model_config)
    classifier = ClassifierModel(model_config)

    classifier.configure_pipeline()
    train_data, val_data = classifier.prepare_data()
    classifier.fit_classifier(train_data)
    classifier.predict(val_data)
