FROM apache/spark-py

USER root

WORKDIR /app

RUN mkdir experiments
RUN chmod -R 765 experiments

ADD /data /app/data
ADD /config /app/config
ADD /src /app/src
ADD requirements.txt /app

RUN pip install -r requirements.txt

ENV SPARK_DRIVER_MEMORY=16G
ENV SPARK_EXECUTOR_CORES=12
ENV SPARK_EXECUTOR_MEMORY=16G
ENV SPARK_WORKER_CORES=12
ENV SPARK_WORKER_MEMORY=16G
