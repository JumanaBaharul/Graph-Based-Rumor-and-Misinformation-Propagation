from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import RumorGraphDataset, collate_graphs
from .model import RumorGCN


def train_one_epoch(
    model: RumorGCN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x = batch["x"].to(device)
        adj = batch["adj"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["y"].to(device).squeeze(-1)

        optimizer.zero_grad()
        logits = model(x, adj, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(
    model: RumorGCN, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["y"].to(device).squeeze(-1)

            logits = model(x, adj, mask)
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).long()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": report["accuracy"],
        "precision_rumor": report["1"]["precision"],
        "recall_rumor": report["1"]["recall"],
        "f1_rumor": report["1"]["f1-score"],
        "precision_verified": report["0"]["precision"],
        "recall_verified": report["0"]["recall"],
        "f1_verified": report["0"]["f1-score"],
        "confusion_matrix": cm.tolist(),
    }


def run_training(
    data_path: Path,
    epochs: int = 80,
    batch_size: int = 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
    save_model_path: Path | None = None,
) -> None:
    device = torch.device(device)
    dataset = RumorGraphDataset(data_path)
    labels = np.array([sample.label.item() for sample in dataset.samples])
    input_dim = dataset[0].x.shape[1]

    skf = StratifiedKFold(n_splits=min(3, len(dataset)), shuffle=True, random_state=42)

    all_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold + 1} ---")
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graphs,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )

        model = RumorGCN(
            in_features=input_dim,
            hidden_dim=64,
            temporal_feature_index=input_dim - 1,
            dropout=0.25,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:03d} - Loss: {loss:.4f}")

        metrics = evaluate(model, test_loader, device)
        all_metrics.append(metrics)
        print("Fold metrics:")
        for key, value in metrics.items():
            if key != "confusion_matrix":
                print(f"  {key}: {value:.4f}")
        print(f"  confusion_matrix: {metrics['confusion_matrix']}")

    print("\n=== Cross-validation summary ===")
    for key in [
        "accuracy",
        "precision_rumor",
        "recall_rumor",
        "f1_rumor",
        "precision_verified",
        "recall_verified",
        "f1_verified",
    ]:
        values = [metrics[key] for metrics in all_metrics]
        print(f"{key}: mean={np.mean(values):.4f} ± {np.std(values):.4f}")

    if save_model_path is not None:
        print("\nTraining final model on the full dataset for deployment...")
        full_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graphs,
        )
        model = RumorGCN(
            in_features=input_dim,
            hidden_dim=64,
            temporal_feature_index=input_dim - 1,
            dropout=0.25,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            loss = train_one_epoch(model, full_loader, optimizer, criterion, device)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"[Full data] Epoch {epoch + 1:03d} - Loss: {loss:.4f}")

        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "config": {
                    "hidden_dim": 64,
                    "dropout": 0.25,
                    "lr": lr,
                    "epochs": epochs,
                },
            },
            save_model_path,
        )
        if dataset.vectorizer is not None:
            vectorizer_path = save_model_path.with_suffix(".vectorizer.pkl")
            joblib.dump(dataset.vectorizer, vectorizer_path)
            print(f"Saved vectorizer to {vectorizer_path}")
        print(f"Saved final model to {save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the temporal-aware rumor GCN model")
    parser.add_argument("--data", type=Path, default=Path("data/sample_graphs.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-model", type=Path, default=None)
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_model_path=args.save_model,
    )
