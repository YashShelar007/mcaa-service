#!/usr/bin/env python3
"""
quantize_dynamic.py
Demonstrates dynamic quantization of the baseline model, then measures size & accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
import torch.quantization as quant
import time

def main():
    # 1. Load baseline model
    model_fp32 = resnet18(num_classes=10)
    model_fp32.load_state_dict(torch.load("baseline_resnet18_cifar.pth"))

    # 2. Dynamic quantization (only affects nn.Linear by default)
    # Note: For bigger effect, you'd do QAT or post-training static quant with conv layers.
    model_int8 = quant.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)

    # Save & check size
    torch.save(model_int8.state_dict(), "resnet18_dynamic_quant.pth")
    print("Saved quantized model to resnet18_dynamic_quant.pth")

    # 3. Evaluate size difference
    import os
    fp32_size = os.path.getsize("baseline_resnet18_cifar.pth") / 1024**2
    int8_size = os.path.getsize("resnet18_dynamic_quant.pth") / 1024**2
    print(f"Baseline model size: {fp32_size:.2f} MB")
    print(f"Dynamic quant model size: {int8_size:.2f} MB")
    print(f"Size reduction: ~{fp32_size/int8_size:.2f}×")

    # 4. Evaluate accuracy
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_int8 = model_int8.to(device)
    test_loader = get_test_loader()
    acc = evaluate(model_int8, test_loader, device)
    print(f"Quantized model accuracy: {acc:.2f}%")

    # 5. Measure inference time on a dummy input
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    # Warm-up
    for _ in range(10):
        _ = model_int8(dummy_input)
    # Timed
    start = time.time()
    for _ in range(50):
        _ = model_int8(dummy_input)
    end = time.time()
    avg_time = (end - start) / 50
    print(f"Avg inference time (1 batch of size=1): {avg_time*1000:.3f} ms")

def get_test_loader():
    transform = T.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    return test_loader

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

if __name__ == "__main__":
    main()
