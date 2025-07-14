#!/usr/bin/env python3
"""
baseline.py

Dual mode:
1) CLI:  python baseline.py <input-model>          # just measures metrics
2) ECS:  runs with MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE env-vars
"""

import os
import sys
import time
import warnings
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from logger import log
from model_loader import auto_load_model

# CIFAR-10 normalization constants
_C10_MEAN = (0.4914, 0.4822, 0.4465)
_C10_STD  = (0.2470, 0.2435, 0.2616)

def get_test_loader(batch_size=128, data_root="/tmp/data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_C10_MEAN, _C10_STD),
    ])
    ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tf
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

def measure_latency(model, device, warmup=10, runs=50):
    model.eval().to(device)
    dummy = torch.randn(1,3,32,32).to(device)
    for _ in range(warmup):
        _ = model(dummy)
    start = time.time()
    for _ in range(runs):
        _ = model(dummy)
    return (time.time() - start) / runs

def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outs = model(imgs)
            _, preds = outs.max(1)
            total += labs.size(0)
            correct += (preds == labs).sum().item()
    return 100.0 * correct / total

def _choose_qengine():
    """Prefer FBGEMM on x86, else QNNPACK."""
    engines = torch.backends.quantized.supported_engines
    if 'fbgemm' in engines:
        return 'fbgemm'
    if 'qnnpack' in engines:
        return 'qnnpack'
    raise RuntimeError(f"No quant engine available; supported={engines}")

# -------------- CLI Mode --------------

def cli_mode():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="Path to a .pt checkpoint (state_dict or pickled/JIT)")
    args = p.parse_args()

    device = torch.device("cpu")
    # 1) pick the best engine
    engine = _choose_qengine()
    torch.backends.quantized.engine = engine
    model  = auto_load_model(args.model_path, device)

    print("Evaluating baseline model metrics…")
    acc     = evaluate(model, get_test_loader(), device)
    latency = measure_latency(model, device)
    size_mb = os.path.getsize(args.model_path) / (1024**2)
    print(
        f"[CLI] acc={acc:.2f}%, size={size_mb:.2f} MB, "
        f"latency={latency*1000:.2f} ms"
    )

# -------------- ECS Mode --------------

def ecs_mode():
    warnings.filterwarnings("ignore")
    import boto3

    bucket  = os.environ["MODEL_BUCKET"]
    key     = os.environ["MODEL_S3_KEY"]
    user    = os.environ["USER_ID"]
    profile = os.environ["PROFILE"]

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    # ─── Keep original filename so auto_load_model sees it ──────────────────
    basename = os.path.basename(key)         
    tmp_path = f"/tmp/{basename}"

    print(f"Downloading baseline model from s3://{bucket}/{key} to {tmp_path} …")
    s3.download_file(bucket, key, tmp_path)
    print("Download complete")

    device = torch.device("cpu")
    # 1) pick the best engine
    engine = _choose_qengine()
    torch.backends.quantized.engine = engine
    model  = auto_load_model(tmp_path, device)

    print("Running evaluation…")
    acc     = evaluate(model, get_test_loader(), device)
    latency = measure_latency(model, device)
    size_mb = os.path.getsize(tmp_path) / (1024**2)
    print(f"[ECS] acc={acc:.2f}%, size={size_mb:.2f} MB, latency={latency*1000:.2f} ms")

    # Re-upload into profile folder
    out_key = key.replace("/baseline/", f"/{profile}/baseline/")
    s3.upload_file(tmp_path, bucket, out_key)
    print(f"Uploaded baseline model to s3://{bucket}/{out_key}")

    # Record to DynamoDB & CSV‐log
    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{basename}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "baseline"},
        "Accuracy":  {"N": f"{acc:.2f}"},
        "SizeMB":    {"N": f"{size_mb:.3f}"},
        "LatencyMS": {"N": f"{latency*1000:.2f}"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
      "script":     "baseline",
      "step":       "baseline",
      "accuracy":   acc,
      "size_mb":    size_mb,
      "latency_ms": latency * 1000.0
    })

    print("Baseline evaluation step complete.")
    # emit JSON so StepFunctions / status-lambda can pick up metrics
    print(json.dumps({
        "accuracy":   acc,
        "size_mb": size_mb,
        "latency_ms": latency * 1000.0
    }))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        ecs_mode()
