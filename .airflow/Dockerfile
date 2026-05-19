FROM apache/airflow:2.10.5-python3.10

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY .airflow/requirements.txt /tmp/airflow-project-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-project-requirements.txt
