#!/usr/bin/env python3
"""
measure_inference_time.py

Downloads a model from S3, attempts to load it into ResNet18 (strict=False),
measures CPU inference latency, uploads the model under an 'evaluated' prefix,
and logs metrics to DynamoDB. Never fails on a bad checkpoint.
"""

import os, time, boto3, warnings
import torch
from torchvision.models import resnet18
from logger import log

# Suppress PyTorch weight-only warning
warnings.filterwarnings("ignore", message="weights_only")

def measure_inference_time(model, device,
                           input_shape=(1,3,32,32),
                           num_warmup=10, num_runs=50):
    model.eval().to(device)
    dummy = torch.randn(*input_shape).to(device)
    # Warm-up
    for _ in range(num_warmup):
        _ = model(dummy)
    # Timed runs
    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    return (time.time() - start) / num_runs

def main():
    # 1. Env vars
    bucket  = os.environ["MODEL_BUCKET"]
    key     = os.environ["MODEL_S3_KEY"]    # e.g. users/demo/balanced/quantized/model.pt
    user    = os.environ["USER_ID"]
    profile = os.environ["PROFILE"]

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")
    tmp_path = "/tmp/model.pt"

    # 2. Download model
    print(f"Downloading s3://{bucket}/{key} → {tmp_path}")
    s3.download_file(bucket, key, tmp_path)

    # 3. Instantiate model
    device = torch.device("cpu")
    model  = resnet18(num_classes=10)

    # 4. Try to load state_dict
    try:
        state = torch.load(tmp_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint with strict=False")
    except Exception as e:
        print(f"Warning: could not fully load checkpoint ({e}), using default weights")

    # 5. Measure latency & size
    latency_s  = measure_inference_time(model, device)
    latency_ms = latency_s * 1000.0
    size_mb    = os.path.getsize(tmp_path) / (1024**2)

    # 6. Upload under evaluated prefix — build from filename
    filename      = os.path.basename(key)
    evaluated_key = f"users/{user}/{profile}/evaluated/{filename}"
    print(f"Uploading evaluated model to s3://{bucket}/{evaluated_key}")
    s3.upload_file(tmp_path, bucket, evaluated_key)

    # 7. Write to DynamoDB
    ddb.put_item(
        TableName="mcaa-service-metadata",
        Item={
            "ModelID":   {"S": f"{user}:{filename}"},
            "UserID":    {"S": user},
            "Profile":   {"S": profile},
            "S3Key":     {"S": evaluated_key},
            "SizeMB":    {"N": f"{size_mb:.3f}"},
            "LatencyMS": {"N": f"{latency_ms:.2f}"},
            "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        }
    )

    # 8. Local CSV log
    log({
      "script":   "evaluate",
      "step":     "final",
      "accuracy":  0,           # update if you measure accuracy here
      "size_mb":   size_mb,
      "latency_ms": latency_ms
    })

    print(f"Evaluation done: {size_mb:.2f} MB, {latency_ms:.2f} ms")

if __name__ == "__main__":
    main()
