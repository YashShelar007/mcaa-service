#!/usr/bin/env python3
"""
distill.py
Teacher = baseline, Student = pruned. Does knowledge distillation to recover accuracy.
Saves 'distilled_resnet18_cifar.pth'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
import torch.nn.functional as F

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Teacher: baseline
    teacher = resnet18(num_classes=10)
    teacher.load_state_dict(torch.load("baseline_resnet18_cifar.pth"))
    teacher.eval().to(device)

    # 2. Student: pruned
    student = resnet18(num_classes=10)
    student.load_state_dict(torch.load("pruned_resnet18_cifar.pth"))
    student = student.to(device)

    # 3. Data
    train_loader, test_loader = get_loaders()

    # 4. Distillation hyperparams
    T = 4.0
    alpha = 0.5
    epochs = 1
    optimizer = optim.Adam(student.parameters(), lr=0.0005)
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            # Hard loss
            hard_loss = criterion_ce(student_logits, labels)
            # Soft distillation loss
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T * T)

            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Distillation] Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")
        acc = evaluate(student, test_loader, device)
        print(f"Test Accuracy after epoch {epoch+1}: {acc:.2f}%")

    # 5. Save
    torch.save(student.state_dict(), "distilled_resnet18_cifar.pth")
    print("Saved KD-distilled model to distilled_resnet18_cifar.pth")

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
