FROM python:3.9-slim

WORKDIR /app

# 1) Install runtime dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Add boto3 here
RUN pip install --no-cache-dir boto3

# 2) Copy your updated evaluator script & logger
COPY modules/compression/prune.py .
COPY modules/compression/logger.py .

# 3) Entrypoint
ENTRYPOINT ["python", "prune.py"]
