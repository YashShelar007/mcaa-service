#!/usr/bin/env python3
"""
train_baseline.py
Train a baseline ResNet18 on CIFAR-10 for demonstration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

def main():
    # Hyperparams
    epochs = 2  # Increase for better accuracy
    batch_size = 64
    lr = 0.001

    # 1. Prepare CIFAR-10
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                transform=T.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 2. Define ResNet18, modify final layer for 10 classes
    model = resnet18(pretrained=False)  # pretrained=True is for ImageNet, not matching CIFAR shape
    model.fc = nn.Linear(model.fc.in_features, 10)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # M1 "Metal" or CPU fallback
    model = model.to(device)

    # 3. Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training loop (minimal)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

        # Quick test each epoch
        acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy after epoch {epoch+1}: {acc:.2f}%")

    # 5. Save the baseline model
    torch.save(model.state_dict(), "baseline_resnet18_cifar.pth")
    print("Saved baseline model to baseline_resnet18_cifar.pth")

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
