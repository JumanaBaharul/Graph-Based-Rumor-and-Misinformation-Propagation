from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import torch

from .data import RumorGraphDataset
from .model import RumorGCN


def load_graph(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_sample(dataset: RumorGraphDataset, graph_id: Optional[str]) -> tuple:
    if graph_id is None:
        return dataset[0]
    for sample in dataset.samples:
        if sample.graph_id == graph_id:
            return sample
    raise ValueError(f"Graph id {graph_id} not found in dataset")


def run_inference(
    model_path: Path,
    vectorizer_path: Path,
    data_path: Path,
    graph_json: Optional[Path] = None,
    graph_id: Optional[str] = None,
    device: str = "cpu",
) -> None:
    dataset = RumorGraphDataset(data_path)
    dataset.vectorizer = joblib.load(vectorizer_path)

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    input_dim = checkpoint["input_dim"]

    model = RumorGCN(
        in_features=input_dim,
        hidden_dim=config["hidden_dim"],
        temporal_feature_index=input_dim - 1,
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if graph_json is not None:
        graph_data = load_graph(graph_json)
        sample = dataset.transform_graph(graph_data)
    else:
        sample = select_sample(dataset, graph_id)

    x = sample.x.unsqueeze(0).to(device)
    adj = sample.adj.unsqueeze(0).to(device)
    mask = torch.ones(1, sample.x.shape[0], device=device)

    with torch.no_grad():
        probability = model.predict(x, adj, mask).item()

    predicted_label = 1 if probability >= 0.5 else 0
    label_name = "Rumor / Misinformation" if predicted_label == 1 else "Verified"

    print(f"Graph ID: {sample.graph_id}")
    print(f"Predicted probability of rumor: {probability:.4f}")
    print(f"Predicted label: {label_name}")

    # Highlight top influential nodes based on activation magnitude
    model.eval()
    with torch.no_grad():
        logits = model.forward(x, adj, mask)
        hidden_repr = model.layer2(
            model.layer1(x, adj, mask), adj, mask
        ).squeeze(0)
    node_scores = hidden_repr.norm(dim=-1).cpu().numpy()
    top_indices = node_scores.argsort()[::-1][:3]
    print("Top contributing posts:")
    for idx in top_indices:
        meta = sample.node_metadata[idx]
        print(f"  Node {idx} ({meta['user']}): score={node_scores[idx]:.3f}\n    text: {meta['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the trained rumor GCN model")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vectorizer", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/sample_graphs.json"))
    parser.add_argument("--graph-json", type=Path, default=None)
    parser.add_argument("--graph-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        vectorizer_path=args.vectorizer,
        data_path=args.data,
        graph_json=args.graph_json,
        graph_id=args.graph_id,
        device=args.device,
    )
