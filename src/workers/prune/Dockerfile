FROM python:3.9-slim

WORKDIR /app

# Install only PyTorch
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the compression script and logger
COPY modules/compression/prune.py .
COPY modules/compression/logger.py .

ENTRYPOINT ["python", "prune.py"]
