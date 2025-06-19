#!/usr/bin/env python3
"""
prune.py
Loads the baseline, prunes 50% of conv weights, fine-tunes briefly, saves 'pruned_resnet18_cifar.pth'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18

def main():
    # 1. Load baseline
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load("baseline_resnet18_cifar.pth"))

    # 2. Prune 50% of conv weights unstructured (example)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.5)
            prune.remove(module, 'weight')  # finalize

    # 3. Fine-tune for a bit to recover accuracy
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    train_loader, test_loader = get_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    epochs = 1  # minimal, increase if you want better recovery

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
        print(f"[Pruning Fine-tune] Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

        acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {acc:.2f}%")

    # 4. Save pruned model
    torch.save(model.state_dict(), "pruned_resnet18_cifar.pth")
    print("Saved pruned model to pruned_resnet18_cifar.pth")

def get_loaders():
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    transform_test = T.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    return train_loader, test_loader

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
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
