"""Interactive Streamlit dashboard for rumor detection demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import streamlit as st
import torch

from src.data import RumorGraphDataset
from src.model import HierarchicalRumorNet, RumorGCN, RumorGAT, build_rumor_model

st.set_page_config(page_title="Rumor Graph Detector", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifacts(
    model_path: str,
    vectorizer_path: str,
    data_path: str,
) -> Tuple[RumorGraphDataset, torch.nn.Module, Dict]:
    """Loads dataset, vectorizer, and model for inference."""

    dataset = RumorGraphDataset(data_path)
    vectorizer = joblib.load(vectorizer_path)
    dataset.rebuild_with_vectorizer(vectorizer)

    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    input_dim = checkpoint.get("input_dim", dataset[0].x.shape[1])

    architecture = config.get("architecture", "temporal_gcn")
    model = build_rumor_model(
        architecture,
        in_features=input_dim,
        hidden_dim=config.get("hidden_dim", 64),
        temporal_feature_index=input_dim - 1,
        dropout=config.get("dropout", 0.3),
        heads=config.get("heads", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return dataset, model, config


def predict_graph(model: torch.nn.Module, sample) -> Tuple[float, torch.Tensor]:
    x = sample.x.unsqueeze(0)
    adj = sample.adj.unsqueeze(0)
    mask = torch.ones(1, sample.x.shape[0])

    with torch.no_grad():
        probability = model.predict(x, adj, mask).item()

    return probability, mask


def explain_nodes(model: torch.nn.Module, sample, mask: torch.Tensor) -> List[Tuple[int, float]]:
    x = sample.x.unsqueeze(0)
    adj = sample.adj.unsqueeze(0)

    with torch.no_grad():
        if isinstance(model, HierarchicalRumorNet):
            h1 = model.layer1(x, adj, mask)
            h2 = model.layer2(h1, adj, mask).squeeze(0)
        elif isinstance(model, RumorGCN) or isinstance(model, RumorGAT):
            h2 = model.layer2(model.layer1(x, adj, mask), adj, mask).squeeze(0)
        else:
            # Generic fallback: use forward hook approximation
            h2 = model.layer2(model.layer1(x, adj, mask), adj, mask).squeeze(0)

    node_scores = h2.norm(dim=-1).cpu().numpy()
    top_indices = node_scores.argsort()[::-1][:3]
    return [(int(idx), float(node_scores[idx])) for idx in top_indices]


def render_graph_summary(sample, probability: float, explanation: List[Tuple[int, float]]) -> None:
    label = "Rumor / Misinformation" if probability >= 0.5 else "Verified"
    st.metric("Predicted label", label, delta=f"rumor probability {probability:.2%}")

    st.subheader("Top contributing posts")
    for idx, score in explanation:
        meta = sample.node_metadata[idx]
        st.markdown(
            f"**Node {idx}** — @{meta['user']} (score={score:.3f})\n\n> {meta['text']}"
        )


def build_custom_graph(posts: List[Dict[str, str]]) -> Dict:
    nodes = []
    edges = []
    for i, post in enumerate(posts):
        nodes.append(
            {
                "id": i,
                "user": post.get("user", f"user_{i}"),
                "text": post.get("text", ""),
                "timestamp": post.get("timestamp", "2023-01-01T00:00:00Z"),
            }
        )
        if i > 0:
            edges.append([0, i])
            edges.append([i - 1, i])
    return {"id": "custom_graph", "label": 0, "nodes": nodes, "edges": edges}


st.title("Temporal Graph Rumor Detector")
st.write(
    "Load a trained checkpoint and explore predictions on curated rumor datasets or your own conversation graph."
)

with st.sidebar:
    st.header("Artifacts")
    default_model = "artifacts/tir_gcn.pt"
    model_path = st.text_input("Model checkpoint", value=default_model)
    vectorizer_path = st.text_input("Vectorizer pickle", value="artifacts/tir_gcn.vectorizer.pkl")
    data_path = st.text_input("Dataset", value="data/sample_graphs.json")

    artifacts_loaded = False
    if Path(model_path).exists() and Path(vectorizer_path).exists() and Path(data_path).exists():
        artifacts_loaded = True
    else:
        st.warning("Update the paths above once you have trained and exported a model.")

if artifacts_loaded:
    dataset, model, config = load_artifacts(model_path, vectorizer_path, data_path)
    st.sidebar.success(f"Loaded architecture: {config.get('architecture', 'temporal_gcn')}")

    graph_ids = [sample.graph_id for sample in dataset]
    selection_mode = st.radio("Inference source", ["Dataset sample", "Upload JSON", "Manual entry"], horizontal=True)

    if selection_mode == "Dataset sample":
        selected_id = st.selectbox("Conversation", graph_ids)
        sample = next(sample for sample in dataset.samples if sample.graph_id == selected_id)
    elif selection_mode == "Upload JSON":
        uploaded = st.file_uploader("Upload conversation graph JSON", type="json")
        if uploaded is not None:
            graph_data = json.load(uploaded)
            sample = dataset.transform_graph(graph_data)
        else:
            st.stop()
    else:
        st.info("Provide posts chronologically. We connect them in a simple cascade for demo purposes.")
        num_posts = st.slider("Number of posts", min_value=2, max_value=8, value=3)
        posts: List[Dict[str, str]] = []
        for i in range(num_posts):
            col1, col2 = st.columns([1, 3])
            with col1:
                user = st.text_input(f"User {i}", value=f"user_{i}")
            with col2:
                text = st.text_area(f"Post {i} text", value=f"Example text {i}", height=80)
            timestamp = st.text_input(
                f"Timestamp {i}", value=f"2023-01-01T00:0{i}:00Z", key=f"ts_{i}"
            )
            posts.append({"user": user, "text": text, "timestamp": timestamp})
        custom_graph = build_custom_graph(posts)
        sample = dataset.transform_graph(custom_graph)

    probability, mask = predict_graph(model, sample)
    explanation = explain_nodes(model, sample, mask)

    left, right = st.columns([2, 1])
    with left:
        render_graph_summary(sample, probability, explanation)
    with right:
        st.subheader("Graph metadata")
        st.json({"graph_id": sample.graph_id, "num_nodes": len(sample.node_metadata), "config": config})

else:
    st.stop()
