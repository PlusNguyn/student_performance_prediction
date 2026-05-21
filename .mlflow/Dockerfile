ARG PYTHON_BASE_IMAGE=python:3.10-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

WORKDIR /mlflow

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --prefer-binary -r requirements.txt

EXPOSE 5000

CMD ["sh", "-c", "mlflow server --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --artifacts-destination /mlflow/artifacts --default-artifact-root mlflow-artifacts:/ --serve-artifacts --host 0.0.0.0 --port 5000"]
