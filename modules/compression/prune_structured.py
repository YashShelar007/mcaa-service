#!/usr/bin/env python3
"""
prune_structured.py

Structured channel pruning on CIFAR-10 models (ResNet/VGG/etc).

Dual mode:
 1) CLI: python prune_structured.py <in_model> <out_model.pt> [--prune_ratio R] [--ft_epochs E]
 2) ECS: runs with MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE,
         optional PRUNE_RATIO, FT_EPOCHS env-vars

Outputs a TorchScript .pt in both modes.
"""
import os
import sys
import time
import json
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as D
import torch_pruning as tp

from model_loader import auto_load_model
from logger import log

# CIFAR-10 normalization
_C10_MEAN = (0.4914, 0.4822, 0.4465)
_C10_STD  = (0.2470, 0.2435, 0.2616)

def get_loaders(batch_size=128):
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_C10_MEAN, _C10_STD),
    ])
    tf_test = T.Compose([
        T.ToTensor(),
        T.Normalize(_C10_MEAN, _C10_STD),
    ])
    train_ds = D.CIFAR10(root="/tmp/data", train=True, download=True, transform=tf_train)
    test_ds  = D.CIFAR10(root="/tmp/data", train=False, download=True, transform=tf_test)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        torch.utils.data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2),
    )

def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds==y).sum().item()
            total   += y.size(0)
    return 100.0 * correct/total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_latency(model, device, runs=30):
    model.eval().to(device)
    dummy = torch.randn(1,3,32,32, device=device)
    # warm-up
    for _ in range(10):
        _ = model(dummy)
    start = time.time()
    for _ in range(runs):
        _ = model(dummy)
    return (time.time() - start)/runs * 1000.0  # ms per forward

def structured_prune(model, prune_ratio, device):
    # build dependency graph
    DG = tp.DependencyGraph().build_dependency(
        model, torch.randn(1,3,32,32, device=device)
    )
    for m in list(model.modules()):
        if not isinstance(m, nn.Conv2d):
            continue
        out_ch = m.weight.shape[0]
        n_prune = int(out_ch * prune_ratio)
        if n_prune < 1:
            continue
        # rank channels by L1 norm
        ranks = m.weight.abs().sum(dim=(1,2,3))
        prune_idxs = torch.argsort(ranks)[:n_prune].tolist()
        # prune those channels
        group = DG.get_pruning_group(
            m,
            tp.prune_conv_out_channels,
            prune_idxs
        )
        group.prune()
    return model

# ── CLI Mode ────────────────────────────────────────────────────────────────
def cli_mode():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("in_model",  help=".pth state_dict or TorchScript .pt")
    p.add_argument("out_model", help="Where to save pruned TorchScript .pt")
    p.add_argument("--prune_ratio","-r", type=float, default=0.5,
                   help="Fraction of channels to remove in each conv")
    p.add_argument("--ft_epochs","-e", type=int,   default=2,
                   help="Fine‐tune epochs after pruning")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders()
    crit = nn.CrossEntropyLoss()

    # 1) Load & baseline
    print("[CLI] Loading model…")
    model = auto_load_model(args.in_model, device)
    acc0    = evaluate(model, test_loader, device)
    params0 = count_parameters(model)
    lat0    = measure_latency(model, device)
    size0   = os.path.getsize(args.in_model)/(1024**2)
    print(f"[CLI] Baseline → acc={acc0:.2f}%, params={params0:,}, "
          f"lat={lat0:.1f}ms, size={size0:.2f}MB")

    # 2) Structured prune
    print(f"[CLI] Pruning {int(args.prune_ratio*100)}% channels…")
    pruned = structured_prune(model, args.prune_ratio, device)

    # 3) Fine‐tune
    print(f"[CLI] Fine‐tuning for {args.ft_epochs} epochs…")
    optimizer = optim.SGD(pruned.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    for ep in range(1, args.ft_epochs+1):
        pruned.train()
        total_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = crit(pruned(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc_t = evaluate(pruned, test_loader, device)
        print(f"[CLI]   Epoch {ep}/{args.ft_epochs} → loss={total_loss/len(train_loader):.4f}, acc={acc_t:.2f}%")

    # 4) JIT‐script & save
    acc1    = evaluate(pruned, test_loader, device)
    params1 = count_parameters(pruned)
    lat1    = measure_latency(pruned, device)
    try:
        jm = torch.jit.script(pruned)
    except Exception:
        jm = torch.jit.trace(pruned, torch.randn(1,3,32,32, device=device))
    torch.jit.save(jm, args.out_model)
    # torch.save(pruned.state_dict(), args.out_model)
    size1 = os.path.getsize(args.out_model)/(1024**2)
    print(f"[CLI] Pruned   → acc={acc1:.2f}%, params={params1:,}, "
          f"lat={lat1:.1f}ms, size={size1:.2f}MB")
    print(f"[CLI] Saved pruned model to {args.out_model}")

# ── ECS Mode ────────────────────────────────────────────────────────────────
def ecs_mode():
    warnings.filterwarnings("ignore")
    import boto3

    bucket       = os.environ["MODEL_BUCKET"]
    key          = os.environ["MODEL_S3_KEY"]
    user         = os.environ["USER_ID"]
    profile      = os.environ["PROFILE"]
    prune_ratio  = float(os.environ.get("PRUNE_RATIO", 0.5))
    ft_epochs    = int(os.environ.get("FT_EPOCHS",   2))

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    basename = os.path.basename(key)
    tmp_in   = f"/tmp/{basename}"
    tmp_out  = f"/tmp/pruned_{basename}"

    print(f"[ECS] Downloading s3://{bucket}/{key} → {tmp_in}")
    s3.download_file(bucket, key, tmp_in)

    device     = torch.device("cpu")
    model      = auto_load_model(tmp_in, device)
    train_loader, test_loader = get_loaders()
    crit       = nn.CrossEntropyLoss()

    # baseline
    acc0    = evaluate(model, test_loader, device)
    params0 = count_parameters(model)
    lat0    = measure_latency(model, device)
    size0   = os.path.getsize(tmp_in)/(1024**2)
    print(f"[ECS] Baseline → acc={acc0:.2f}%, param={params0}, lat={lat0:.1f}ms, size={size0:.2f}MB")

    # structured prune
    print(f"[ECS] Pruning {int(prune_ratio*100)}% of output channels in each conv…")
    pruned = structured_prune(model, prune_ratio, device)

    # fine‐tune
    print(f"→ Fine-tuning for {ft_epochs} epochs…")
    optimizer = optim.SGD(pruned.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    for ep in range(1, ft_epochs+1):
        pruned.train()
        total_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = crit(pruned(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc_t = evaluate(pruned, test_loader, device)
        print(f"  Epoch {ep}/{ft_epochs} → loss={total_loss/len(train_loader):.4f}, acc={acc_t:.2f}%")
    
    # JIT‐script & save
    acc1    = evaluate(pruned, test_loader, device)
    lat1    = measure_latency(pruned, device)
    params1 = count_parameters(pruned)
    try:
        jm = torch.jit.script(pruned)
    except Exception:
        jm = torch.jit.trace(pruned, torch.randn(1,3,32,32, device=device))
    torch.jit.save(jm, tmp_out)
    # torch.save(pruned.state_dict(), tmp_out)
    size1 = os.path.getsize(tmp_out)/(1024**2)
    print(f"[ECS] Baseline → acc={acc1:.2f}%, param={params1}, lat={lat1:.1f}ms, size={size1:.2f}MB")

    # upload
    out_key = f"users/{user}/{profile}/pruned/{basename}"
    s3.upload_file(tmp_out, bucket, out_key)
    print(f"[ECS] Uploaded pruned model → s3://{bucket}/{out_key}")
    # record metadata
    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{basename}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "prune"},
        "Accuracy":  {"N": f"{acc1:.2f}"},
        "SizeMB":    {"N": f"{size1:.3f}"},
        "LatencyMS": {"N": f"{lat1:.2f}"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
        "script":    "prune_structured",
        "step":      "prune_structured",
        "accuracy":  acc1,
        "size_mb":   size1,
        "latency_ms": lat1,
    })

    # emit JSON for Step Functions
    print(json.dumps({
        "accuracy":   acc1,
        "size_bytes": os.path.getsize(tmp_out)
    }))

    print("[ECS] Prune step complete.")

if __name__ == "__main__":
    if len(sys.argv)>1 and os.path.isfile(sys.argv[1]):
        cli_mode()
    else:
        ecs_mode()
