FROM python:3.9-slim

WORKDIR /app

# 1) install git so we can clone at build time
RUN apt-get update \
 && apt-get install -y git \
 && rm -rf /var/lib/apt/lists/*

# 2) clone the CIFAR-hub repo and add a dummy requirements.txt
RUN git clone https://github.com/chenyaofo/pytorch-cifar-models.git /opt/pytorch-cifar-models \
 && echo "" > /opt/pytorch-cifar-models/requirements.txt

# 3) install the repo
RUN pip install --no-cache-dir /opt/pytorch-cifar-models

# 4) install any other Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt boto3

# 5) copy your loader + logger
COPY modules/compression/model_loader.py .
COPY modules/compression/logger.py .

ENTRYPOINT ["python", "model_loader.py"]
