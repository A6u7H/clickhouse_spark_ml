import os
import yaml
import pandas as pd
import configparser

from clickhouse_driver import Client
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    def __init__(self, config):
        self.config = config
        with open(self.config["CONST"]["PATH"], 'r') as stream:
            self.project_params = yaml.safe_load(stream)

    def get_create_table_query(
        self,
        table_name: str,
        data_types: Dict[str, str]
    ):
        query = f"CREATE TABLE IF NOT EXISTS {table_name}"
        columns = []
        for k, v in data_types.items():
            if v == "int64":
                v = "Int64"
            elif v == "int32":
                v = "Int64"
            elif v == "object":
                v = "String"
            elif v == "float64":
                v = "Float"
            columns.append(f"{k} {v}")
        columns_name = ", ".join(columns)
        columns_name = "(" + columns_name + ")"
        query = query + columns_name + " Engine = Memory"
        return query

    def upload_data(self):
        data = pd.read_csv(
            self.config["DATA"]["DATAPATH"],
            sep="\t",
            nrows=self.config.getint("CLICKHOUSE", "UPLOAD_LIMIT")
        )
        data = data.rename(columns=self.project_params["rename"])
        data = data[self.project_params["columns"]]
        data = data.fillna(0)
        client = Client(
            host=self.config["CLICKHOUSE"]["HOST"],
            port=self.config["CLICKHOUSE"]["PORT"],
            database=os.environ["CLICKHOUSE_DB"],
            user=os.environ["CLICKHOUSE_USER"],
            password=os.environ["CLICKHOUSE_PASSWORD"]
        )

        table_name = self.config["CLICKHOUSE"]["TABLE_NAME"]
        query = self.get_create_table_query(table_name, dict(data.dtypes))

        client.execute(query)
        client.execute(f"INSERT INTO {table_name} VALUES", data.to_dict('records'))

    def load(self):
        client = Client(
            host=self.config["CLICKHOUSE"]["HOST"],
            port=self.config["CLICKHOUSE"]["PORT"],
            database=os.environ["CLICKHOUSE_DB"],
            user=os.environ["CLICKHOUSE_USER"],
            password=os.environ["CLICKHOUSE_PASSWORD"]
        )

        table_name = self.config["CLICKHOUSE"]["TABLE_NAME"]
        nrows = self.config.getint("CLICKHOUSE", "DOWNLOAD_LIMIT")
        data = client.execute(f"SELECT * FROM {table_name} limit {nrows}")
        df = pd.DataFrame(
            data=data,
            columns=self.project_params["columns"]
        )
        df.to_parquet(self.config["CLICKHOUSE"]["DATA_SAVE_PATH"], index=False)

    def upload_prediction(self, data):
        client = Client(
            host=self.config["CLICKHOUSE"]["HOST"],
            port=self.config["CLICKHOUSE"]["PORT"],
            database=os.environ["CLICKHOUSE_DB"],
            user=os.environ["CLICKHOUSE_USER"],
            password=os.environ["CLICKHOUSE_PASSWORD"]
        )
        table_name = "Cluster"
        query = self.get_create_table_query(table_name, dict(data.dtypes))
        client.execute(query)
        client.execute(f"INSERT INTO {table_name} VALUES", data.to_dict('records'))


if __name__ == "__main__":
    config_path = "./config/config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    uploader = DataLoader(config)
    uploader.upload_data()
    uploader.load()
