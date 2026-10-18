"""Entrainement du CNN de detection de pneumonie (NORMAL / PNEUMONIA).

Ce script reproduit l'entrainement realise dans le notebook
`notebooks/Chest_XRay_training.ipynb` et produit le fichier de poids
`weights/cnn_pneumonia_model.pth` utilise par `src/vision_module.py`.

Dataset : "Chest X-Ray Images (Pneumonia)" (Paul Mooney, Kaggle).
Structure attendue :

    data/chest_xray/
        train/NORMAL, train/PNEUMONIA
        val/NORMAL,   val/PNEUMONIA
        test/NORMAL,  test/PNEUMONIA

Utilisation (depuis la racine du projet) :

    python -m src.train_vision --data-dir data/chest_xray

Remarques :
- Le CNN est importe depuis `vision_module.py` : une seule definition du modele,
  donc l'entrainement et l'inference ne peuvent pas diverger.
- Les transformations ci-dessous doivent rester identiques a celles de
  `MedicalVisionModule` (redimensionnement 128x128, niveaux de gris, normalisation).
- Le jeu de validation Kaggle ne contient que 16 images : il sert au suivi, pas
  a la selection du meilleur modele. Les poids sauvegardes sont ceux de la
  derniere epoch, comme dans le notebook.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.vision_module import PneumoniaCNN

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Poids de classes : une erreur sur NORMAL coute 2x plus qu'une erreur sur PNEUMONIA,
# pour limiter les faux positifs (le modele a tendance a sur-predire PNEUMONIA).
CLASS_WEIGHTS = [2.0, 1.0]


def set_seed(seed):
    """Fixe les graines aleatoires pour rendre l'entrainement reproductible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def build_loaders(data_dir, batch_size):
    transform = build_transforms()
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        if dataset.classes != CLASS_NAMES:
            raise ValueError(
                f"Classes inattendues dans '{split}': {dataset.classes} (attendu: {CLASS_NAMES})"
            )
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    return loaders


def evaluate(model, loader, device, criterion=None):
    """Retourne (loss moyenne ou None, etiquettes reelles, predictions)."""
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader) if criterion is not None else None
    return avg_loss, np.array(all_labels), np.array(all_preds)


def train(data_dir, epochs, batch_size, learning_rate, seed, output_path):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Appareil utilise : {device}")

    loaders = build_loaders(data_dir, batch_size)
    model = PneumoniaCNN().to(device)

    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(loaders["train"])
        val_loss, val_labels, val_preds = evaluate(model, loaders["val"], device, criterion)
        val_acc = float((val_labels == val_preds).mean())
        print(
            f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f} "
            f"- val loss: {val_loss:.4f} - val acc: {val_acc:.2%}"
        )

    # Evaluation finale sur le jeu de test.
    _, test_labels, test_preds = evaluate(model, loaders["test"], device)
    accuracy = float((test_labels == test_preds).mean())
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall_pneumonia = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"\n[TEST] Accuracy : {accuracy:.2%}")
    print(f"[TEST] Matrice de confusion (lignes = reel, colonnes = predit) :\n{cm}")
    print(f"[TEST] Rappel PNEUMONIA : {recall_pneumonia:.2%} | Specificite NORMAL : {specificity:.2%}")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=3))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[INFO] Poids sauvegardes dans {output_path}")

    metrics_path = os.path.splitext(output_path)[0] + "_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "class_weights": CLASS_WEIGHTS,
                "test_accuracy": accuracy,
                "confusion_matrix": cm.tolist(),
                "recall_pneumonia": recall_pneumonia,
                "specificity_normal": specificity,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Metriques sauvegardees dans {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrainement du CNN NORMAL / PNEUMONIA")
    parser.add_argument("--data-dir", default="data/chest_xray", help="Dossier contenant train/ val/ test/")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="weights/cnn_pneumonia_model.pth")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        output_path=args.output,
    )
