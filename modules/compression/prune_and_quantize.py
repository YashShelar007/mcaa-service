#!/usr/bin/env python3
"""
prune_and_quantize.py

Runs prune_search to pick the best pruned model, then FX‐quant or JIT‐quant to hit your
bitwidth target (8‐bit static with FX or JIT), final save as TorchScript .pt.
"""
import os
import sys
import time
import warnings
import json

import torch
import torch.nn as nn

from logger import log
from model_loader import auto_load_model

# reuse our prune‐search logic
import prune_search

# reuse our quant helpers
from quantize import (
    apply_fx_static_quant,
    apply_jit_quant,
    get_test_loader,
)

from torchvision import transforms, datasets

# CIFAR-10 normalization constants
_C10_MEAN = (0.4914, 0.4822, 0.4465)
_C10_STD  = (0.2470, 0.2435, 0.2616)


def get_test_loader_local(batch_size=128, data_root="/tmp/data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_C10_MEAN, _C10_STD),
    ])
    ds = datasets.CIFAR10(root=data_root, train=False,
                          download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                       shuffle=False, num_workers=2)


def evaluate(model, loader, device):
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            _, pred = out.max(1)
            correct += (pred == labs).sum().item()
            total   += labs.size(0)
    return 100.0 * correct / total


def measure_file_bytes(path):
    return os.path.getsize(path)


def measure_latency(model, device, warmup=5, runs=20):
    model.to(device).eval()
    dummy = torch.randn(1,3,32,32, device=device)
    for _ in range(warmup):
        _ = model(dummy)
    t0 = time.time()
    for _ in range(runs):
        _ = model(dummy)
    return (time.time() - t0) / runs  # seconds


def _run_prune_then_quant(state_source, acc_tol, max_bytes, bitwidth):
    # 1) pick best pruned model
    best_mod, _, _, _ = prune_search._run_prune_loop(
        state_source, acc_tol, max_bytes
    )

    # 2) apply FX‐static or JIT‐static quant
    loader = get_test_loader_local()
    if bitwidth == 8:
        if isinstance(best_mod, torch.jit.RecursiveScriptModule):
            quant_mod = apply_jit_quant(best_mod, loader)
        else:
            quant_mod = apply_fx_static_quant(best_mod, loader)
    else:
        # fallback: FP32 or FP16 half-precision
        quant_mod = best_mod.half() if bitwidth == 16 else best_mod

    # 3) evaluate & measure
    device = torch.device("cpu")
    acc_q  = evaluate(quant_mod, loader, device)

    tmp = "/tmp/quant_pt.pt"
    try:
        jm = torch.jit.script(quant_mod)
    except:
        jm = torch.jit.trace(quant_mod, torch.randn(1,3,32,32))
    torch.jit.save(jm, tmp)

    size_q = measure_file_bytes(tmp)
    lat_q  = measure_latency(quant_mod, device)
    return quant_mod, acc_q, size_q, lat_q


def cli_mode(in_model, out_model, acc_tol, size_limit_mb, bitwidth):
    warnings.filterwarnings("ignore")
    max_bytes = size_limit_mb * 1024**2

    quant_mod, acc_q, size_q, lat_q = _run_prune_then_quant(
        in_model, acc_tol, max_bytes, bitwidth
    )

    # save final TorchScript
    try:
        jm = torch.jit.script(quant_mod)
    except:
        jm = torch.jit.trace(quant_mod, torch.randn(1,3,32,32))
    torch.jit.save(jm, out_model)

    print(f"[CLI] Final → acc={acc_q:.2f}%, "
          f"size={size_q/1024**2:.2f}MiB, lat={lat_q*1000:.2f}ms")
    print(json.dumps({
        "accuracy":    acc_q,
        "size_bytes":  size_q,
        "latency_ms":  lat_q*1000.0
    }))


def ecs_mode():
    warnings.filterwarnings("ignore")
    import boto3

    bucket        = os.environ["MODEL_BUCKET"]
    key_pruned    = os.environ["PRUNED_S3_KEY"]
    user          = os.environ["USER_ID"]
    profile       = os.environ["PROFILE"]
    acc_tol       = float(os.environ.get("ACC_TOL",        0.0))
    size_limit_mb = float(os.environ.get("SIZE_LIMIT_MB",  0.0))
    bitwidth      = int(os.environ.get("BITWIDTH",        8))

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    basename = os.path.basename(key_pruned)
    tmp_in   = f"/tmp/{basename}"
    tmp_out  = f"/tmp/quant_{basename}"

    print(f"[ECS] Downloading pruned model → {tmp_in}")
    s3.download_file(bucket, key_pruned, tmp_in)

    quant_mod, acc_q, size_q, lat_q = _run_prune_then_quant(
        tmp_in, acc_tol, size_limit_mb*1024**2, bitwidth
    )

    print("[ECS] Saving TorchScript…")
    try:
        jm = torch.jit.script(quant_mod)
    except:
        jm = torch.jit.trace(quant_mod, torch.randn(1,3,32,32))
    torch.jit.save(jm, tmp_out)

    out_key = key_pruned.replace(f"/{profile}/pruned/", f"/{profile}/quantized/")
    print(f"[ECS] Uploading to s3://{bucket}/{out_key}")
    s3.upload_file(tmp_out, bucket, out_key)

    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{basename}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "prune_and_quantize"},
        "Accuracy":  {"N": f"{acc_q:.2f}"},
        "SizeMB":    {"N": f"{size_q/1024**2:.3f}"},
        "LatencyMS": {"N": f"{lat_q*1000:.2f}"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
      "script":     "prune_and_quantize",
      "step":       "prune_and_quantize",
      "accuracy":   acc_q,
      "size_mb":    size_q/1024**2,
      "latency_ms": lat_q*1000.0
    })

    print(json.dumps({
        "accuracy":   acc_q,
        "size_bytes": size_q,
        "latency_ms": lat_q*1000.0
    }))
    print("[ECS] Prune and Quantize complete.")


if __name__ == "__main__":
    if os.environ.get("MODEL_BUCKET"):
        ecs_mode()
    else:
        import argparse
        p = argparse.ArgumentParser(prog="prune_and_quantize.py")
        p.add_argument("in_model")
        p.add_argument("out_model")
        p.add_argument("--acc_tol",       type=float, default=0.0)
        p.add_argument("--size_limit_mb", type=float, default=0.0)
        p.add_argument("--bitwidth",      type=int,   default=8)
        args = p.parse_args()
        cli_mode(args.in_model, args.out_model,
                 args.acc_tol, args.size_limit_mb, args.bitwidth)
