#!/usr/bin/env python3
"""
distill_kd.py

Knowledge‐distillation on CIFAR-10 models.

Dual mode:
 1) CLI: python distill_kd.py <teacher> <student_init> <out_distilled.pth>
 2) ECS: runs with MODEL_BUCKET, BASELINE_S3_KEY, PRUNED_S3_KEY, USER_ID, PROFILE,
         optional TEMPERATURE, ALPHA, EPOCHS env-vars
"""
import os
import sys
import time
import warnings
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from logger import log
from model_loader import auto_load_model
from torchvision import transforms, datasets

# CIFAR-10 normalization constants
_C10_MEAN = (0.4914, 0.4822, 0.4465)
_C10_STD  = (0.2470, 0.2435, 0.2616)

def get_loaders(batch_size=64, data_root="/tmp/data"):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_C10_MEAN, _C10_STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_C10_MEAN, _C10_STD),
    ])
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    test_ds  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_test)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        torch.utils.data.DataLoader(test_ds,  batch_size=128,       shuffle=False, num_workers=2),
    )

def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outs = model(imgs)
            _, preds = outs.max(1)
            correct += (preds == labs).sum().item()
            total   += labs.size(0)
    return 100.0 * correct / total

# ── CLI Mode ────────────────────────────────────────────────────────────────
def cli_mode():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("baseline",    help="Teacher checkpoint (.pth/JIT/module)")
    p.add_argument("student",     help="Student checkpoint (pruned)")
    p.add_argument("out",         help="Where to write distilled model (.pt)")
    p.add_argument("--temperature","-T", type=float, default=4.0,
                   help="Softmax temperature")
    p.add_argument("--alpha",      "-a", type=float, default=0.9,
                   help="Weight for hard-label loss")
    p.add_argument("--epochs",     "-e", type=int,   default=5,
                   help="Number of distillation epochs")
    args = p.parse_args()

    device       = torch.device("cpu")
    teacher      = auto_load_model(args.baseline, device)
    student      = auto_load_model(args.student,  device)
    T, alpha, E  = args.temperature, args.alpha, args.epochs

    train_loader, test_loader = get_loaders()
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(student.parameters(), lr=5e-4)

    print(f"[CLI] Distilling: T={T}, α={alpha}, epochs={E}")
    for epoch in range(1, E+1):
        student.train()
        total_loss = 0.0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            with torch.no_grad():
                tlog = teacher(imgs)
            slog = student(imgs)
            loss_h = criterion(slog, labs)
            loss_s = F.kl_div(
                F.log_softmax(slog / T, dim=1),
                F.softmax(tlog / T, dim=1),
                reduction="batchmean"
            ) * (T * T)
            loss = alpha * loss_h + (1 - alpha) * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate(student, train_loader, device)
        test_acc  = evaluate(student, test_loader,  device)
        print(
            f"[CLI] Epoch {epoch}/{E} "
            f"loss={avg_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_acc={test_acc:.2f}%"
        )

    final_acc = evaluate(student, test_loader, device)
    # save full model (state_dict)
    try:
        jm = torch.jit.script(student)
    except Exception:
        jm = torch.jit.trace(student, torch.randn(1,3,32,32, device=device))
    torch.jit.save(jm, args.out)
    #torch.save(student.state_dict(), args.out)
    # emit JSON for Step Functions / parsing
    size_bytes = os.path.getsize(args.out)
    print(json.dumps({"accuracy": final_acc, "size_mb": size_bytes}))
    print(f"[CLI] Distill KD done: acc={final_acc:.2f}%, saved to {args.out}")

# ── ECS Mode ────────────────────────────────────────────────────────────────
def ecs_mode():
    warnings.filterwarnings("ignore")
    import boto3

    bucket       = os.environ["MODEL_BUCKET"]
    baseline_key = os.environ["BASELINE_S3_KEY"]
    pruned_key   = os.environ["PRUNED_S3_KEY"]
    user         = os.environ["USER_ID"]
    profile      = os.environ["PROFILE"]
    T            = float(os.environ.get("TEMPERATURE", 4.0))
    alpha        = float(os.environ.get("ALPHA",       0.9))
    epochs       = int(os.environ.get("EPOCHS",       5))

    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    base_teacher = os.path.basename(baseline_key)
    base_student = os.path.basename(pruned_key)
    tmp_teacher  = f"/tmp/teacher_{base_teacher}"
    tmp_student  = f"/tmp/student_{base_student}"
    tmp_out      = f"/tmp/distilled_{base_student}"

    print(f"[ECS] Downloading teacher & student…")
    print(f"[ECS] Downloading s3://{bucket}/{baseline_key} → {tmp_teacher}")
    s3.download_file(bucket, baseline_key, tmp_teacher)
    print(f"[ECS] Downloading s3://{bucket}/{pruned_key} → {tmp_student}")
    s3.download_file(bucket, pruned_key,   tmp_student)

    device       = torch.device("cpu")
    teacher      = auto_load_model(tmp_teacher, device)
    student      = auto_load_model(tmp_student, device)
    train_loader, test_loader = get_loaders()
    criterion   = nn.CrossEntropyLoss()

    # baseline
    accT    = evaluate(teacher, test_loader, device)
    print(f"[ECS] Teacher → acc={accT:.2f}%")
    # prune
    accS    = evaluate(student, test_loader, device)
    print(f"[ECS] Student → acc={accS:.2f}%")


    optimizer   = optim.Adam(student.parameters(), lr=5e-4)

    print(f"[ECS] Distilling: T={T}, α={alpha}, epochs={epochs}")
    for epoch in range(1, epochs+1):
        student.train()
        total_loss = 0.0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            with torch.no_grad():
                tlog = teacher(imgs)
            slog = student(imgs)
            loss_h = criterion(slog, labs)
            loss_s = F.kl_div(
                F.log_softmax(slog / T, dim=1),
                F.softmax(tlog / T, dim=1),
                reduction="batchmean"
            ) * (T * T)
            loss = alpha * loss_h + (1 - alpha) * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(student, test_loader, device)
        print(f"[ECS] Epoch {epoch}/{epochs} avg loss {avg_loss:.4f}, test_acc={test_acc:.2f}%")

    final_acc = evaluate(student, test_loader, device)
    try:
        jm = torch.jit.script(student)
    except Exception:
        jm = torch.jit.trace(student, torch.randn(1,3,32,32, device=device))
    torch.jit.save(jm, tmp_out)
    # torch.save(student.state_dict(), tmp_out)
    print(f"[ECS] Distill KD complete: Student acc={final_acc:.2f}%")

    # upload
    out_key = f"users/{user}/{profile}/distilled/{base_student}"
    s3.upload_file(tmp_out, bucket, out_key)
    size_bytes = os.path.getsize(tmp_out)
    print(f"[ECS] Uploaded distilled model to s3://{bucket}/{out_key}")

    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{base_student}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "distill_kd"},
        "Accuracy":  {"N": f"{final_acc:.2f}"},
        "SizeMB":    {"N": f"{size_bytes/1024**2:.3f}"},
        "LatencyMS": {"N": "0"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
      "script":    "distill_kd",
      "step":      "distill_kd",
      "accuracy":  final_acc,
      "size_mb":   size_bytes/1024**2,
      "latency_ms":0,
    })

    # Emit JSON for Step Functions
    print(json.dumps({"accuracy": final_acc, "size_mb": size_bytes/1024**2}))

    print("[ECS] Distill KD step complete.")

if __name__ == "__main__":
    # CLI if first arg is a file path
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        cli_mode()
    else:
        ecs_mode()
