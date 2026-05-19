FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /mlflow

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["sh", "-c", "mlflow server --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --artifacts-destination /mlflow/artifacts --default-artifact-root mlflow-artifacts:/ --serve-artifacts --host 0.0.0.0 --port 5000"]
