from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class GraphSample:
    """Container for a single conversation graph."""

    x: torch.Tensor  # Node features [num_nodes, num_features]
    adj: torch.Tensor  # Normalized adjacency with self loops [num_nodes, num_nodes]
    label: torch.Tensor  # Graph label (rumor=1, verified=0)
    node_metadata: List[Dict]
    graph_id: str


class RumorGraphDataset(torch.utils.data.Dataset):
    """Loads rumor conversation graphs and prepares tensors for GCN models."""

    def __init__(self, json_path: str | Path, max_features: int = 100) -> None:
        self.path = Path(json_path)
        raw_graphs = self._load_graphs(self.path)
        self.vectorizer: TfidfVectorizer | None = None
        self.samples: List[GraphSample] = self._vectorize_graphs(raw_graphs, max_features)

    @staticmethod
    def _load_graphs(path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _vectorize_graphs(self, graphs: List[Dict], max_features: int) -> List[GraphSample]:
        # Gather all node texts for TF-IDF vocabulary learning
        all_texts: List[str] = []
        for graph in graphs:
            all_texts.extend(node["text"] for node in graph["nodes"])

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        text_features = vectorizer.fit_transform(all_texts).toarray().astype(np.float32)
        self.vectorizer = vectorizer

        samples: List[GraphSample] = []
        cursor = 0
        for graph in graphs:
            nodes = graph["nodes"]
            num_nodes = len(nodes)
            node_text_feats = text_features[cursor : cursor + num_nodes]
            cursor += num_nodes

            adj = self._build_normalized_adjacency(graph["edges"], num_nodes)
            structural_feats = self._structural_features(nodes, graph["edges"], num_nodes)

            features = np.concatenate([node_text_feats, structural_feats], axis=1)

            samples.append(
                GraphSample(
                    x=torch.from_numpy(features),
                    adj=adj,
                    label=torch.tensor([graph["label"]], dtype=torch.float32),
                    node_metadata=nodes,
                    graph_id=graph["id"],
                )
            )
        return samples

    @staticmethod
    def _build_normalized_adjacency(edges: List[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(edges)
        adj = nx.to_numpy_array(graph, dtype=np.float32)

        # Add self loops
        adj += np.eye(num_nodes, dtype=np.float32)

        # Symmetric normalization D^{-1/2} A D^{-1/2}
        degrees = adj.sum(axis=1)
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = np.diag(d_inv_sqrt)
        normalized_adj = d_mat @ adj @ d_mat
        return torch.from_numpy(normalized_adj)

    @staticmethod
    def _structural_features(
        nodes: List[Dict], edges: List[Tuple[int, int]], num_nodes: int
    ) -> np.ndarray:
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(edges)

        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)

        # Temporal feature: minutes since source node (node 0 assumed source)
        timestamps = np.array([
            np.datetime64(node["timestamp"]) for node in nodes
        ])
        deltas = ((timestamps - timestamps[0]) / np.timedelta64(1, "m")).astype(np.float32)
        if deltas.max() > 0:
            deltas = deltas / deltas.max()
        else:
            deltas = np.zeros_like(deltas, dtype=np.float32)

        struct_feats = np.zeros((num_nodes, 3), dtype=np.float32)
        for idx in range(num_nodes):
            struct_feats[idx, 0] = degree_centrality.get(idx, 0.0)
            struct_feats[idx, 1] = betweenness_centrality.get(idx, 0.0)
            struct_feats[idx, 2] = deltas[idx]
        return struct_feats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]

    def transform_graph(self, graph: Dict) -> GraphSample:
        """Vectorizes a new conversation graph using the fitted vocabulary."""

        if self.vectorizer is None:
            raise ValueError("The dataset vectorizer has not been initialized.")

        nodes = graph["nodes"]
        num_nodes = len(nodes)
        texts = [node["text"] for node in nodes]
        node_text_feats = self.vectorizer.transform(texts).toarray().astype(np.float32)

        adj = self._build_normalized_adjacency(graph["edges"], num_nodes)
        structural_feats = self._structural_features(nodes, graph["edges"], num_nodes)
        features = np.concatenate([node_text_feats, structural_feats], axis=1)

        return GraphSample(
            x=torch.from_numpy(features),
            adj=adj,
            label=torch.tensor([graph.get("label", -1)], dtype=torch.float32),
            node_metadata=nodes,
            graph_id=graph.get("id", "inference_graph"),
        )


def collate_graphs(batch: List[GraphSample]) -> Dict[str, torch.Tensor]:
    """Pads graphs in the batch for mini-batch processing."""

    max_nodes = max(sample.x.shape[0] for sample in batch)
    feature_dim = batch[0].x.shape[1]

    x_batch = torch.zeros(len(batch), max_nodes, feature_dim, dtype=torch.float32)
    adj_batch = torch.zeros(len(batch), max_nodes, max_nodes, dtype=torch.float32)
    mask_batch = torch.zeros(len(batch), max_nodes, dtype=torch.float32)
    labels = torch.stack([sample.label for sample in batch])

    metadata: List[Dict[str, List]] = []

    for i, sample in enumerate(batch):
        num_nodes = sample.x.shape[0]
        x_batch[i, :num_nodes] = sample.x
        adj_batch[i, :num_nodes, :num_nodes] = sample.adj
        mask_batch[i, :num_nodes] = 1.0
        metadata.append({"graph_id": sample.graph_id, "nodes": sample.node_metadata})

    return {"x": x_batch, "adj": adj_batch, "mask": mask_batch, "y": labels, "metadata": metadata}
