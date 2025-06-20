#!/usr/bin/env python3
"""
distill.py

Dual mode:
1) CLI:  python distill.py <baseline.pth> <pruned.pth> <out.pth>
2) ECS:  MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE
"""

import os, sys, time, warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from logger import log

# -------------- Helpers --------------

def get_loaders(batch_size=64, data_root="/tmp/data"):
    from torchvision import transforms, datasets
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test  = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    return (
      torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2),
      torch.utils.data.DataLoader(test_ds, batch_size=128,        shuffle=False, num_workers=2),
    )

def evaluate(model, loader, device):
    model.eval().to(device)
    correct, total = 0, 0
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
    p.add_argument("teacher")
    p.add_argument("student")
    p.add_argument("out")
    args = p.parse_args()

    device = torch.device("cpu")
    teacher = resnet18(num_classes=10)
    student = resnet18(num_classes=10)
    teacher.load_state_dict(torch.load(args.teacher, map_location=device), strict=False)
    student.load_state_dict(torch.load(args.student, map_location=device), strict=False)

    # Distillation
    T, alpha = 4.0, 0.5
    opt = optim.Adam(student.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_loaders()

    for epoch in range(1):
        print(f"Distill epoch {epoch+1}")
        student.train()
        total = 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            with torch.no_grad():
                tlog = teacher(imgs)
            slog = student(imgs)
            loss_h = criterion(slog, labs)
            loss_s = F.kl_div(
                F.log_softmax(slog/T,dim=1),
                F.softmax(tlog/T,dim=1),
                reduction="batchmean") * (T*T)
            loss = alpha*loss_h + (1-alpha)*loss_s
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

    acc = evaluate(student, test_loader, device)
    torch.save(student.state_dict(), args.out)
    print(f"CLI distill done: acc={acc:.2f}%, saved to {args.out}")

# -------------- ECS Mode --------------

def ecs_mode():
    warnings.filterwarnings("ignore")
    bucket = os.environ["MODEL_BUCKET"]
    baseline_key  = os.environ["BASELINE_S3_KEY"]
    pruned_key    = os.environ["PRUNED_S3_KEY"]
    user   = os.environ["USER_ID"]
    profile= os.environ["PROFILE"]

    import boto3
    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    tmp_base = "/tmp/base.pt"
    tmp_prun = "/tmp/pruned.pt"
    tmp_out  = "/tmp/distilled.pt"

    print("Downloading baseline & pruned models…")
    s3.download_file(bucket, baseline_key, tmp_base)
    s3.download_file(bucket, pruned_key, tmp_prun)
    print("Download complete")

    device  = torch.device("cpu")
    teacher = resnet18(num_classes=10)
    student = resnet18(num_classes=10)
    teacher.load_state_dict(torch.load(tmp_base, map_location=device), strict=False)
    student.load_state_dict(torch.load(tmp_prun, map_location=device), strict=False)

    # Distill hyperparams
    T, alpha = 4.0, 0.5
    opt       = optim.Adam(student.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_loaders()

    print("Running knowledge distillation…")
    student.train()
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        with torch.no_grad(): tlog = teacher(imgs)
        slog = student(imgs)
        loss_h = criterion(slog, labs)
        loss_s = F.kl_div(
            F.log_softmax(slog/T,dim=1),
            F.softmax(tlog/T,dim=1),
            reduction="batchmean") * (T*T)
        loss = alpha*loss_h + (1-alpha)*loss_s
        opt.zero_grad(); loss.backward(); opt.step()

    acc = evaluate(student, test_loader, device)
    torch.save(student.state_dict(), tmp_out)
    print(f"Distillation complete: acc={acc:.2f}%")

    # Upload distilled
    out_key = pruned_key.replace(f"/{profile}/pruned/", f"/{profile}/distilled/")
    s3.upload_file(tmp_out, bucket, out_key)
    print(f"Uploaded distilled model to {out_key}")

    # Log to DynamoDB
    size_mb = os.path.getsize(tmp_out)/(1024**2)
    # after you’ve computed `out_key` …
    base_name = os.path.basename(out_key)
    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID": {"S": f"{user}:{base_name}"},
        "UserID":  {"S": user},
        "Profile": {"S": profile},
        "Step":    {"S": "distill"},
        "Accuracy": {"N": f"{acc:.2f}"},
        "SizeMB":   {"N": f"{size_mb:.3f}"},
        "LatencyMS":{"N": "0"},
        "Timestamp":{"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )

    log({
      "script":    "distill",
      "step":      "distill",
      "accuracy":  acc,
      "size_mb":   size_mb,
      "latency_ms": 0,
    })

    print("Distill step complete.")

# -------------- Entrypoint --------------

if __name__ == "__main__":
    if len(sys.argv)>1:
        cli_mode()
    else:
        ecs_mode()
