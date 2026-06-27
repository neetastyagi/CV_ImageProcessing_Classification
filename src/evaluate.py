"""Example evaluation script for CV_Project (PyTorch)

Usage example:
python src/evaluate.py --model models/best_model.pth --data_dir data/<dataset> --batch_size 32
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset")
    parser.add_argument("--model", required=True, help="Path to saved model .pth file")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (ImageFolder format)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    classes = checkpoint.get("classes")
    # Build a model that matches the saved checkpoint
    import torch.nn as nn
    from torchvision import models
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), classes


def build_loader(data_dir, batch_size, num_workers):
    test_dir = os.path.join(data_dir, "val")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset.classes


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, classes = load_model(args.model, device)
    loader, dataset_classes = build_loader(args.data_dir, args.batch_size, args.num_workers)

    y_true, y_pred = evaluate(model, loader, device)
    acc = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0
    print(f"Accuracy: {acc:.4f} ({len(y_true)} samples)")

    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion(y_true, y_pred, classes, cm_path)
    print(f"Saved confusion matrix to {cm_path}")
