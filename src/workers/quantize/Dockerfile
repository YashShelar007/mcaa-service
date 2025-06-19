FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY modules/compression/quantize_dynamic.py .
COPY modules/compression/logger.py .

ENTRYPOINT ["python", "quantize_dynamic.py"]
