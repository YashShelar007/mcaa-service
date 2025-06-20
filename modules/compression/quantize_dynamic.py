#!/usr/bin/env python3
"""
quantize_dynamic.py

Dual mode:
1) CLI:  python quantize_dynamic.py <input-model> <output-model>
2) ECS:  runs with MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE env-vars
"""

import os, sys, time, warnings
import torch
import torch.nn as nn
import torch.quantization as quant
from torchvision.models import resnet18
from torchvision import transforms, datasets
from logger import log

# -------------------- Helpers --------------------

def get_test_loader(batch_size=128, data_root="/tmp/data"):
    transform = transforms.ToTensor()
    ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outs = model(imgs)
            _, pred = outs.max(1)
            total += labs.size(0)
            correct += (pred==labs).sum().item()
    return 100.0*correct/total

def measure_latency(model, device, input_shape=(1,3,32,32), warmup=10, runs=50):
    model.eval().to(device)
    dummy = torch.randn(*input_shape).to(device)
    for _ in range(warmup): _ = model(dummy)
    start = time.time()
    for _ in range(runs): _ = model(dummy)
    return (time.time()-start)/runs

# -------------------- CLI Mode --------------------

def cli_mode():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("in_model")
    p.add_argument("out_model")
    args = p.parse_args()

    device = torch.device("cpu")
    model_fp32 = resnet18(num_classes=10)
    state = torch.load(args.in_model, map_location=device, weights_only=False)
    model_fp32.load_state_dict(state, strict=False)

    print("Applying dynamic quantization…")
    model_int8 = quant.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)

    torch.save(model_int8.state_dict(), args.out_model)
    print(f"Saved quantized model to {args.out_model}")

    fp32_size = os.path.getsize(args.in_model)/(1024**2)
    int8_size = os.path.getsize(args.out_model)/(1024**2)
    print(f"Size: {fp32_size:.2f}→{int8_size:.2f} MB ({fp32_size/int8_size:.2f}×)")

    acc = evaluate(model_int8, get_test_loader(), device)
    latency_s = measure_latency(model_int8, device)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Latency: {latency_s*1000:.2f} ms")

# ------------------- ECS Mode --------------------

def ecs_mode():
    warnings.filterwarnings("ignore")
    bucket = os.environ["MODEL_BUCKET"]
    key    = os.environ["MODEL_S3_KEY"]      # users/demo/baseline/model.pt
    user   = os.environ["USER_ID"]
    profile= os.environ["PROFILE"]

    import boto3
    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    tmp_in  = "/tmp/in.pt"
    tmp_out = "/tmp/quantized.pt"

    print("Downloading model…")
    s3.download_file(bucket, key, tmp_in)
    print("Download complete")

    device = torch.device("cpu")
    model_fp32 = resnet18(num_classes=10)
    state = torch.load(tmp_in, map_location=device, weights_only=False)
    model_fp32.load_state_dict(state, strict=False)

    print("Applying dynamic quantization…")
    model_int8 = quant.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)

    torch.save(model_int8.state_dict(), tmp_out)
    print("Quantized model saved")
    """
    fp32_size = os.path.getsize(tmp_in)/(1024**2)
    int8_size = os.path.getsize(tmp_out)/(1024**2)
    size_red  = fp32_size/int8_size
    print(f"Size: {fp32_size:.2f}→{int8_size:.2f} MB ({size_red:.2f}×)")

    test_loader = get_test_loader(data_root="/tmp/data")
    acc = evaluate(model_int8, test_loader, device)
    latency_s = measure_latency(model_int8, device)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Latency: {latency_s*1000:.2f} ms")
    """
    # Only measure size + dummy-input latency (skip full accuracy pass)
    fp32_size = os.path.getsize(tmp_in)/(1024**2)
    int8_size = os.path.getsize(tmp_out)/(1024**2)
    print(f"Size: {fp32_size:.2f}→{int8_size:.2f} MB")

    # Fast latency test on dummy input
    latency_s = measure_latency(model_int8, device, input_shape=(1,3,32,32), warmup=5, runs=10)
    print(f"Latency (dummy input): {latency_s*1000:.2f} ms")

    # Upload to S3 under profile-specific prefix
    if "/pruned/" in key:
        out_key = key.replace(f"/{profile}/pruned/", f"/{profile}/quantized/")
    else:
        out_key = key.replace(f"/{profile}/distilled/", f"/{profile}/quantized/")
    s3.upload_file(tmp_out, bucket, out_key)

    print(f"Uploaded quantized model to {out_key}")

    # Record to DynamoDB
    ddb.put_item(
        TableName="mcaa-service-metadata",
        Item={
            "ModelID":   {"S": f"{user}:{os.path.basename(key)}"},
            "UserID":    {"S": user},
            "Profile":   {"S": profile},
            "Step":      {"S": "quantize"},
            # "Accuracy":  {"N": f"{acc:.2f}"},
            "SizeMB":    {"N": f"{int8_size:.3f}"},
            "LatencyMS": {"N": f"{latency_s*1000:.2f}"},
            "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        }
    )
    log({
      "script":   "quantize",
      "step":     "quantize",
      # "accuracy": acc,
      "size_mb":  int8_size,
      "latency_ms": latency_s*1000.0
    })

    print("Quantization step complete.")

# ---------------- Dispatcher ----------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_mode()
    else:
        ecs_mode()
