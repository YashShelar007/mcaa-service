#!/usr/bin/env python3
"""
train_baseline.py

Dual mode:
1) CLI:  python train_baseline.py <input-model>          # just measures metrics
2) ECS:  runs with MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE env-vars
"""

import os, sys, time, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms, datasets
from logger import log

# -------------- Helpers --------------

def get_test_loader(batch_size=128, data_root="/tmp/data"):
    transform = transforms.ToTensor()
    ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

def measure_latency(model, device, warmup=10, runs=50):
    model.eval().to(device)
    dummy = torch.randn(1,3,32,32).to(device)
    for _ in range(warmup): _ = model(dummy)
    start = time.time()
    for _ in range(runs): _ = model(dummy)
    return (time.time()-start)/runs

def evaluate(model, loader, device):
    model.eval().to(device)
    correct, total = 0,0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outs = model(imgs)
            _,pred = outs.max(1)
            total += labs.size(0)
            correct += (pred==labs).sum().item()
    return 100.0*correct/total

# -------------- CLI Mode --------------

def cli_mode():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model_path", help="Path to a .pth state_dict to evaluate")
    args = p.parse_args()

    device = torch.device("cpu")
    model  = resnet18(num_classes=10)
    state  = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)

    print("Evaluating baseline model metrics…")
    acc      = evaluate(model, get_test_loader(), device)
    latency  = measure_latency(model, device)
    size_mb  = os.path.getsize(args.model_path)/(1024**2)
    print(f"Baseline metrics: acc={acc:.2f}%, size={size_mb:.2f} MB, latency={latency*1000:.2f} ms")

# -------------- ECS Mode --------------

def ecs_mode():
    warnings.filterwarnings("ignore")
    # Env
    bucket = os.environ["MODEL_BUCKET"]
    key    = os.environ["MODEL_S3_KEY"]    # e.g. users/demo/baseline/model.pt
    user   = os.environ["USER_ID"]
    profile= os.environ["PROFILE"]

    import boto3
    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    tmp_in  = "/tmp/baseline.pt"

    print("Downloading baseline model…")
    s3.download_file(bucket, key, tmp_in)
    print("Download complete")

    device = torch.device("cpu")
    model  = resnet18(num_classes=10)
    state  = torch.load(tmp_in, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    print("Loaded baseline checkpoint (strict=False)")

    acc     = evaluate(model, get_test_loader(), device)
    latency = measure_latency(model, device)
    size_mb = os.path.getsize(tmp_in)/(1024**2)
    print(f"Baseline eval: acc={acc:.2f}%, size={size_mb:.2f} MB, latency={latency*1000:.2f} ms")

    # Upload (no change) into profile-specific folder
    out_key = key.replace("/baseline/", f"/{profile}/baseline/")
    s3.upload_file(tmp_in, bucket, out_key)
    print(f"Uploaded baseline model to {out_key}")

    # DynamoDB record
    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{os.path.basename(key)}"},
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
      "script":    "baseline",
      "step":      "baseline",
      "accuracy":  acc,
      "size_mb":   size_mb,
      "latency_ms": latency*1000.0
    })

    print("Baseline step complete.")

# -------------- Entrypoint --------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        ecs_mode()
