"""
Trains a lightweight 3-class classifier (person / hazard / neither) to
replace the hardcoded heuristic router, using transfer learning on
MobileNetV3-Small. Uses class-weighted loss to correct for the dataset
imbalance (5000 hazard vs 559 person vs 362 neither in your case).
"""

import argparse
import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

CLASSES = ["hazard", "neither", "person"]  # alphabetical - matches ImageFolder ordering


def build_model(num_classes=3):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for param in model.features.parameters():
        param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(dataset, num_classes):
    counts = Counter([label for _, label in dataset.samples])
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/light_classifier.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    print("Detected classes (alphabetical order):", full_dataset.classes)
    assert full_dataset.classes == CLASSES, (
        f"Expected classes {CLASSES}, got {full_dataset.classes} - "
        f"check your data/ folder names match exactly."
    )

    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weights(full_dataset, len(CLASSES)).to(device)
    print("Class weights (to correct imbalance):", dict(zip(CLASSES, class_weights.tolist())))

    model = build_model(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": CLASSES}, args.out)
            print(f"  -> saved new best model (val_acc={val_acc:.3f}) to {args.out}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved at: {args.out}")


if __name__ == "__main__":
    main()