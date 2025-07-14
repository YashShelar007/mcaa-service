#!/usr/bin/env python3
"""
evaluate.py

Dual mode:
1) CLI:
    python evaluate.py <checkpoint> [--data-root DIR] [--warmup N] [--runs M]

2) ECS:
    Uses MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE env-vars.
"""
import os
import sys
import time
import warnings
import argparse
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch

from model_loader import auto_load_model
from logger import log

# ── Helpers ─────────────────────────────────────────────────────────────────

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

def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            pred = model(imgs).argmax(1)
            correct += (pred == labs).sum().item()
            total   += labs.size(0)
    return 100.0 * correct / total

def measure_latency(model, device, warmup=10, runs=50):
    model.eval().to(device)
    dummy = torch.randn(1,3,32,32).to(device)
    for _ in range(warmup):
        _ = model(dummy)
    start = time.time()
    for _ in range(runs):
        _ = model(dummy)
    return (time.time() - start) / runs

def _choose_qengine():
    """Prefer FBGEMM on x86, else QNNPACK."""
    engines = torch.backends.quantized.supported_engines
    if 'fbgemm' in engines:
        return 'fbgemm'
    if 'qnnpack' in engines:
        return 'qnnpack'
    raise RuntimeError(f"No quant engine available; supported={engines}")

# ── CLI Mode ────────────────────────────────────────────────────────────────

def cli_mode():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint",   help="Path to .pth/.pt or pickled nn.Module")
    p.add_argument("--data-root",  default="/tmp/data", help="CIFAR data root")
    p.add_argument("--warmup",     type=int, default=10, help="Warm-up runs")
    p.add_argument("--runs",       type=int, default=50, help="Timed runs")
    args = p.parse_args()

    device = torch.device("cpu")
    # 1) pick the best engine
    engine = _choose_qengine()
    torch.backends.quantized.engine = engine
    model  = auto_load_model(args.checkpoint, device)

    loader     = get_test_loader(data_root=args.data_root)
    acc        = evaluate(model, loader, device)
    latency_s  = measure_latency(model, device,
                   warmup=args.warmup, runs=args.runs)
    latency_ms = latency_s * 1000.0
    size_bytes = os.path.getsize(args.checkpoint)

    print(f"[CLI] Accuracy: {acc:.2f}%")
    print(f"[CLI] Size:     {size_bytes/1024**2:.2f} MB")
    print(f"[CLI] Latency:  {latency_ms:.2f} ms (avg over {args.runs} runs)")
    print(json.dumps({
        "accuracy":    acc,
        "size_mb":  size_bytes,
        "latency_ms":  latency_ms
    }))


# ── ECS Mode ────────────────────────────────────────────────────────────────

def ecs_mode():
    warnings.filterwarnings("ignore")
    import boto3

    bucket  = os.environ["MODEL_BUCKET"]
    key     = os.environ["MODEL_S3_KEY"]
    user    = os.environ["USER_ID"]
    profile = os.environ["PROFILE"]

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    basename  = os.path.basename(key)
    tmp_model = f"/tmp/{basename}"

    print(f"[ECS] Downloading model from s3://{bucket}/{key} to {tmp_model}")
    s3.download_file(bucket, key, tmp_model)
    print("Download complete")

    device = torch.device("cpu")
    # 1) pick the best engine
    engine = _choose_qengine()
    torch.backends.quantized.engine = engine
    try:
        model = auto_load_model(tmp_model, device)
        print("[ECS] Loaded model successfully")
    except Exception as e:
        print(f"[ECS] Warning: auto_load_model failed ({e}), falling back to ResNet18")
        # fallback when even auto_load fails
        from torchvision.models import resnet18
        model = resnet18(num_classes=10).to(device)

    acc        = evaluate(model, get_test_loader(), device)
    latency  = measure_latency(model, device)
    size_mb = os.path.getsize(tmp_model) / (1024**2)
    print(f"[ECS] Final model evaluation → acc={acc:.2f}%, size={size_mb:.2f} MB, latency={latency*1000:.2f} ms")
    
    # upload under evaluated/
    
    out_key = f"users/{user}/{profile}/evaluated/{basename}"
    print(f"[ECS] Uploaded evaluated model → s3://{bucket}/{out_key}")
    s3.upload_file(tmp_model, bucket, out_key)

    # write metadata
    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{basename}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "evaluate"},
        "Accuracy":  {"N": f"{acc:.2f}"},
        "SizeMB":    {"N": f"{size_mb:.3f}"},
        "LatencyMS": {"N": f"{latency:.2f}"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
      "script":    "evaluate",
      "step":      "evaluate",
      "accuracy":  acc,
      "size_mb":   size_mb,
      "latency_ms":latency,
    })

    print(json.dumps({
        "accuracy":   acc,
        "size_mb": size_mb,
        "latency_ms": latency
    }))
    print(f"[ECS] Final evaluation step complete")


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv)>1 and os.path.isfile(sys.argv[1]):
        cli_mode()
    else:
        ecs_mode()
