# src/workers/baseline/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY modules/compression/train_baseline.py .
COPY modules/compression/logger.py .

ENTRYPOINT ["python", "train_baseline.py"]
