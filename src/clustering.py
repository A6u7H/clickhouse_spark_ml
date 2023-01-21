import yaml
import logging
import configparser

from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterModel:
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

    def prepare_data(self):
        df_datamart = self.spark.read.parquet(
            self.config["CLUSTERING"]["INPUT_FILE"]
        ).repartition(self.config.getint("CLUSTERING", "NUM_PARTITIONS"))
        assemble = VectorAssembler(
            inputCols=self.project_params["columns"],
            outputCol='features'
        )
        assembled_data = assemble.transform(df_datamart)
        return assembled_data

    def create_cluster(self, assembled_data):
        evaluator = ClusteringEvaluator(
            featuresCol='standardized',
            metricName='silhouette',
            distanceMeasure='squaredEuclidean'
        )
        scale = StandardScaler(
            inputCol='features',
            outputCol='standardized'
        )

        kmeans_model = KMeans(
            k=self.config.getint("CLUSTERING", "N_CLUSTERS"),
            seed=42,
            featuresCol='standardized',
            predictionCol='prediction'
        )

        data_scale = scale.fit(assembled_data).transform(assembled_data)
        model = kmeans_model.fit(data_scale)
        transform_data = model.transform(data_scale)
        evaluation_score = evaluator.evaluate(transform_data)
        return transform_data, evaluation_score

    def load_data_to_clickhouse(
        self,
        transform_data
    ):
        df_datamart = self.spark.read.parquet(
            self.config["CLUSTERING"]["INPUT_FILE"]
        ).repartition(self.config.getint("CLUSTERING", "NUM_PARTITIONS"))
        df_datamart_pd = df_datamart.toPandas()
        df_datamart_pd["cluster"] = transform_data.select("prediction").toPandas().values.flatten()
        self.loader.upload_prediction(df_datamart_pd)


if __name__ == "__main__":
    config_path = "./config/config.ini"
    model_config = configparser.ConfigParser()
    model_config.read(config_path)

    loader = DataLoader(model_config)
    model = ClusterModel(model_config)

    assembled_data = model.prepare_data()
    transform_data, evaluation_score = model.create_cluster(assembled_data)
    logger.info(f"EVALUATION SCORE: {evaluation_score}")
    model.load_data_to_clickhouse(transform_data)

# conf = SparkConf()
# sc = SparkContext(conf=conf)
# spark = SparkSession(sc)

# df_datamart = spark.read.parquet(
#     model_config["CLUSTERING"]["INPUT_FILE"]
# ).repartition(model_config.getint("CLUSTERING", "NUM_PARTITIONS"))

# assemble = VectorAssembler(inputCols=COLUMNS, outputCol='features')
# assembled_data = assemble.transform(df_datamart)

# evaluator = ClusteringEvaluator(
#     featuresCol='standardized',
#     metricName='silhouette',
#     distanceMeasure='squaredEuclidean'
# )

# scale = StandardScaler(
#     inputCol='features',
#     outputCol='standardized'
# )

# kmeans_model = KMeans(
#     k=model_config.getint("CLUSTERING", "N_CLUSTERS"),
#     seed=42,
#     featuresCol='standardized',
#     predictionCol='prediction'
# )

# data_scale = scale.fit(assembled_data).transform(assembled_data)
# model = kmeans_model.fit(data_scale)
# transform_data = model.transform(data_scale)

# evaluation_score = evaluator.evaluate(transform_data)
# logger.info(f"EVALUATION SCORE: {evaluation_score}")

# df_datamart_pd = df_datamart.toPandas()
# df_datamart_pd["cluster"] = transform_data.select("prediction").toPandas().values.flatten()
# loader.upload_prediction(df_datamart_pd)
