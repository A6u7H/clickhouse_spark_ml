import pandas as pd
import configparser

from clickhouse_driver import Client
from typing import Dict

from consts import COLUMNS, RENAME


class DataLoader:
    def __init__(self, config):
        self.config = config

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
        data = data.rename(columns=RENAME)
        data = data[COLUMNS]
        data = data.fillna(0)

        client = Client(
            host=self.config["CLICKHOUSE"]["HOST"],
            port=self.config["CLICKHOUSE"]["PORT"]
        )

        table_name = self.config["CLICKHOUSE"]["TABLE_NAME"]
        query = self.get_create_table_query(table_name, dict(data.dtypes))

        client.execute(query)
        client.execute(f"INSERT INTO {table_name} VALUES", data.to_dict('records'))

    def load(self, table_name=None, save_path=None, extra_columns=None):
        client = Client(
            host=self.config["CLICKHOUSE"]["HOST"],
            port=self.config["CLICKHOUSE"]["PORT"]
        )
        if table_name is None:
            table_name = self.config["CLICKHOUSE"]["TABLE_NAME"]
        nrows = self.config.getint("CLICKHOUSE", "DOWNLOAD_LIMIT")
        data = client.execute(f"SELECT * FROM {table_name} limit {nrows}")
        columns = COLUMNS + extra_columns if extra_columns is not None else COLUMNS
        df = pd.DataFrame(data=data, columns=columns)
        if save_path is None:
            df.to_parquet(self.config["CLICKHOUSE"]["DATA_SAVE_PATH"], index=False)
        else:
            df.to_parquet(save_path, index=False)

    def upload_prediction(self, data):
        client = Client(
            host=self.config["CLICKHOUSE"]["HOST"],
            port=self.config["CLICKHOUSE"]["PORT"]
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
