"""Example training script for CV_Project (PyTorch)

Usage example:
python src/train.py --data_dir data/<dataset> --epochs 10 --batch_size 32 --output_dir models/
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier on an image dataset")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (ImageFolder format)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    return parser.parse_args()


def build_dataloaders(data_dir, batch_size, num_workers):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes, device):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, classes = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = build_model(len(classes), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / (len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} - loss: {epoch_loss:.4f} - val_acc: {val_acc:.4f} - time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({"model_state_dict": model.state_dict(), "classes": classes}, save_path)
            print(f"Saved best model (val_acc={best_val_acc:.4f}) to {save_path}")

    # Final save
    torch.save({"model_state_dict": model.state_dict(), "classes": classes}, os.path.join(args.output_dir, "final_model.pth"))


if __name__ == "__main__":
    args = parse_args()
    train(args)
