#!/usr/bin/env python3
"""
prune.py

Dual mode:
1) CLI:  python prune.py <local-model> <output-path>
2) ECS:  runs with MODEL_BUCKET, MODEL_S3_KEY, USER_ID, PROFILE env-vars
"""

import os, sys, time, warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune_utils
import torchvision.transforms as T
import torchvision.datasets as D
from torchvision.models import resnet18
from logger import log

# -------------------- Helpers --------------------

def get_loaders(batch_size=64):
    transform_train = T.Compose([T.RandomCrop(32, padding=4),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor()])
    transform_test  = T.ToTensor()

    train_ds = D.CIFAR10(root='/tmp/data', train=True, download=True, transform=transform_train)
    test_ds  = D.CIFAR10(root='/tmp/data', train=False, download=True, transform=transform_test)
    return (torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2))

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

# -------------------- CLI Mode --------------------

def cli_mode():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("in_model", help="Path to baseline .pth")
    p.add_argument("out_model", help="Path to write pruned .pth")
    args = p.parse_args()

    device = torch.device("cpu")
    model  = resnet18(num_classes=10)
    model.load_state_dict(torch.load(args.in_model, map_location=device), strict=False)

    # Prune 50% of CONV weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune_utils.l1_unstructured(m, name="weight", amount=0.5)
            prune_utils.remove(m, "weight")

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=5e-4)

    # Fine‐tune 1 epoch
    model.to(device).train()
    total_loss = 0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        opt.zero_grad()
        loss = criterion(model(imgs), labs)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    acc = evaluate(model, test_loader, device)
    print(f"[CLI] Prune finetune loss={total_loss/len(train_loader):.4f}, acc={acc:.2f}%")

    torch.save(model.state_dict(), args.out_model)
    print(f"[CLI] Saved pruned model to {args.out_model}")

# ------------------- ECS Mode --------------------

def ecs_mode():
    warnings.filterwarnings("ignore")
    # 1. Env
    bucket = os.environ["MODEL_BUCKET"]
    key    = os.environ["MODEL_S3_KEY"]      # e.g. users/demo/baseline/model.pt
    user   = os.environ["USER_ID"]
    profile= os.environ["PROFILE"]

    import boto3
    s3  = boto3.client("s3")
    ddb = boto3.client("dynamodb")

    tmp_in  = "/tmp/in.pt"
    tmp_out = "/tmp/pruned.pt"

    # 2. Download
    print("Downloading model from S3...")
    s3.download_file(bucket, key, tmp_in)
    print("Model downloaded to", tmp_in)

    # 3. Load & prune
    print("Loading model into ResNet18 and applying 50 structured pruning...")
    device = torch.device("cpu")
    model  = resnet18(num_classes=10)
    state  = torch.load(tmp_in, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune_utils.l1_unstructured(m, name="weight", amount=0.5)
            prune_utils.remove(m, "weight")
    print("Pruning complete")

    # 4. Finetune
    print("Starting fine-tuning (1 epoch)...")
    train_loader, test_loader = get_loaders()
    model.to(device).train()
    opt = optim.Adam(model.parameters(), lr=5e-4)
    total_loss = 0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(imgs), labs)
        loss.backward(); opt.step()
        total_loss += loss.item()
    acc = evaluate(model, test_loader, device)
    print(f"Fine-tune done. Validation accuracy: {acc:.2f}%")

    # 5. Save & upload
    torch.save(model.state_dict(), tmp_out)
    out_key = key.replace(f"/{profile}/baseline/", f"/{profile}/pruned/")
    s3.upload_file(tmp_out, bucket, out_key)

    # 6. Log to DynamoDB
    size_mb    = os.path.getsize(tmp_out)/(1024**2)
    ddb.put_item(
      TableName="mcaa-service-metadata",
      Item={
        "ModelID":   {"S": f"{user}:{os.path.basename(key)}"},
        "UserID":    {"S": user},
        "Profile":   {"S": profile},
        "Step":      {"S": "prune"},
        "Accuracy":  {"N": f"{acc:.2f}"},
        "SizeMB":    {"N": f"{size_mb:.3f}"},
        "LatencyMS": {"N": "0"},
        "Timestamp": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
      }
    )
    log({
      "script":   "prune",
      "step":     "prune",
      "accuracy": acc,
      "size_mb":  size_mb,
      "latency_ms": 0,
    })

    print(f"All done! Prune step complete: acc={acc:.2f}%, size={size_mb:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv)>1: cli_mode()
    else:             ecs_mode()
