FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 ESP_MODEL_DIR=/app/models
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && useradd --create-home --uid 10001 appuser
COPY models ./models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "esp_pdm.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS tracking
USER root
COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt && mkdir /tracking && chown appuser:appuser /tracking
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2)"]
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////tracking/mlflow.db", "--artifacts-destination", "/tracking/artifacts", "--serve-artifacts", "--allowed-hosts", "localhost:5000,127.0.0.1:5000,mlflow:5000"]
