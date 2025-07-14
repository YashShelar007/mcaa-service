#!/usr/bin/env python3
"""
prune_search.py

Structured‐pruning search loop with fine‐tuning.

Dual mode:
  1) CLI:  python prune_search.py <in_model.pth> <out_model.pt>
             [--acc_tol 5.0] [--size_limit_mb 5.0] [--ft_epochs 2]
  2) ECS:  runs with MODEL_BUCKET, PRUNED_S3_KEY, USER_ID, PROFILE,
             ACC_TOL, SIZE_LIMIT_MB, FT_EPOCHS env-vars

Tries pruning at 10%,20%,…90%, fine‐tuning each candidate for `ft_epochs`,
and selects the smallest model within the accuracy tolerance and size limit.
"""
import os
import sys
import time
import warnings
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as D

from model_loader import auto_load_model
from logger import log

# structured prune helper from your prune_structured.py
from prune_structured import structured_prune

# CIFAR-10 normalization
_C10_MEAN = (0.4914, 0.4822, 0.4465)
_C10_STD  = (0.2470, 0.2435, 0.2616)


def get_test_loader(batch_size=128):
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(_C10_MEAN, _C10_STD),
    ])
    ds = D.CIFAR10(root="/tmp/data", train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2
    )


def get_train_loader(batch_size=64):
    tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_C10_MEAN, _C10_STD),
    ])
    ds = D.CIFAR10(root="/tmp/data", train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=2
    )


def evaluate(model, loader, device):
    model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            _, pred = model(imgs).max(1)
            correct += (pred == labs).sum().item()
            total   += labs.size(0)
    return 100.0 * correct / total


def measure_file_bytes(path):
    return os.path.getsize(path)


def script_and_save(model, path):
    """JIT‐script & save a model to `path`."""
    try:
        jm = torch.jit.script(model)
    except Exception:
        jm = torch.jit.trace(model, torch.randn(1,3,32,32))
    torch.jit.save(jm, path)


def _run_prune_loop(state_source, acc_tol, max_bytes, ft_epochs):
    device   = torch.device("cpu")
    base     = auto_load_model(state_source, device)
    orig_acc = evaluate(base, get_test_loader(), device)
    print(f"[prune_search] Original → acc={orig_acc:.2f}%")

    best = {"model": None, "acc": 0.0, "size": None, "pct": 0.0}

    for pct in [i/10 for i in range(1,10)]:
        # 1) load fresh model
        m = auto_load_model(state_source, device)
        # 2) structured prune
        m = structured_prune(m, pct, device)

        # 3) fine‐tune for `ft_epochs`
        opt  = optim.Adam(m.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for ep in range(1, ft_epochs+1):
            m.train()
            total_loss = 0.0
            for imgs, labs in get_train_loader():
                imgs, labs = imgs.to(device), labs.to(device)
                opt.zero_grad()
                loss = crit(m(imgs), labs)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(get_train_loader())
            print(f"[prune_search]   prune={int(pct*100)}% FT epoch {ep}/{ft_epochs} loss={avg_loss:.4f}")

        # 4) evaluate & measure size
        acc2 = evaluate(m, get_test_loader(), device)
        tmp  = "/tmp/cand.pt"
        script_and_save(m, tmp)
        sz2  = measure_file_bytes(tmp)

        print(f"[prune_search] {int(pct*100)}% → acc={acc2:.2f}%, size={sz2/1024**2:.2f}MiB")
        if acc2 >= orig_acc - acc_tol and (max_bytes == 0 or sz2 <= max_bytes):
            if best["size"] is None or sz2 < best["size"]:
                best.update(model=m, acc=acc2, size=sz2, pct=pct)

    # fallback if no candidate met criteria
    if best["model"] is None:
        print("[prune_search] fallback → 50% prune")
        m = auto_load_model(state_source, device)
        m = structured_prune(m, 0.5, device)

        # fine‐tune fallback
        opt  = optim.Adam(m.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for ep in range(1, ft_epochs+1):
            m.train()
            total_loss = 0.0
            for imgs, labs in get_train_loader():
                imgs, labs = imgs.to(device), labs.to(device)
                opt.zero_grad()
                loss = crit(m(imgs), labs)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(get_train_loader())
            print(f"[prune_search]   fallback FT epoch {ep}/{ft_epochs} loss={avg_loss:.4f}")

        acc_f = evaluate(m, get_test_loader(), device)
        tmp   = "/tmp/fallback.pt"
        script_and_save(m, tmp)
        sz_f  = measure_file_bytes(tmp)
        best.update(model=m, acc=acc_f, size=sz_f, pct=0.5)

    return best["model"], best["acc"], best["size"], best["pct"]


def cli_mode(in_path, out_path, acc_tol, size_limit_mb, ft_epochs):
    print(f"[CLI] prune_search: {in_path} → {out_path}")
    max_bytes = size_limit_mb * 1024**2
    m, acc, sz, pct = _run_prune_loop(in_path, acc_tol, max_bytes, ft_epochs)
    print(f"[CLI] selected {int(pct*100)}% → acc={acc:.2f}%, size={sz/1024**2:.2f}MiB")

    script_and_save(m, out_path)
    print(f"[CLI] saved to {out_path}")
    print(json.dumps({"accuracy": acc, "size_bytes": sz}))


def ecs_mode():
    warnings.filterwarnings("ignore")
    import boto3

    bucket        = os.environ["MODEL_BUCKET"]
    key           = os.environ["PRUNED_S3_KEY"]
    user          = os.environ["USER_ID"]
    profile       = os.environ["PROFILE"]
    acc_tol       = float(os.environ.get("ACC_TOL",        0.0))
    size_limit_mb = float(os.environ.get("SIZE_LIMIT_MB",  0.0))
    ft_epochs     = int(os.environ.get("FT_EPOCHS",       2))

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    basename = os.path.basename(key)
    tmp_in   = f"/tmp/{basename}"
    tmp_out  = f"/tmp/prune_search_{basename}"

    print(f"[ECS] Downloading s3://{bucket}/{key} → {tmp_in}")
    s3.download_file(bucket, key, tmp_in)

    m, acc, sz, pct = _run_prune_loop(tmp_in, acc_tol, size_limit_mb*1024**2, ft_epochs)

    script_and_save(m, tmp_out)
    s3.upload_file(tmp_out, bucket, key)
    print(f"[ECS] Uploaded prune_search model to s3://{bucket}/{key}")

    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{basename}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "prune_search"},
        "Accuracy":  {"N": f"{acc:.2f}"},
        "SizeMB":    {"N": f"{sz/1024**2:.3f}"},
        "LatencyMS": {"N": "0"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
      "script":    "prune_search",
      "step":      "prune_search",
      "accuracy":  acc,
      "size_mb":   sz/1024**2,
      "latency_ms":0
    })

    print(json.dumps({"accuracy": acc, "size_bytes": sz}))
    print("[ECS] Prune Search done.")


if __name__ == "__main__":
    if os.environ.get("MODEL_BUCKET"):
        ecs_mode()
    else:
        p = argparse.ArgumentParser()
        p.add_argument("in_model")
        p.add_argument("out_model")
        p.add_argument("--acc_tol",       type=float, default=0.0)
        p.add_argument("--size_limit_mb", type=float, default=0.0)
        p.add_argument("--ft_epochs","-e",type=int,   default=2,
                       help="Fine‐tune epochs after each prune step")
        args = p.parse_args()
        cli_mode(args.in_model, args.out_model,
                 args.acc_tol, args.size_limit_mb, args.ft_epochs)
