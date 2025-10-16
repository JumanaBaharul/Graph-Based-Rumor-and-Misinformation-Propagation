# Graph-Based Rumor and Misinformation Propagation

This repository contains a full pipeline for analysing how rumors propagate on social media graphs using a temporal-aware Graph Convolutional Network (GCN). The project includes:

- A curated mini-dataset of conversation graphs (`data/sample_graphs.json`).
- Deep learning training scripts for cross-validated evaluation and model export.
- Inference utilities that highlight influential posts.
- Presentation and demo assets to help you showcase the novelty of the project.

The approach introduces **Temporal Influence Re-weighted GCN (TIR-GCN)**, which augments graph convolutions with propagation speed awareness so the model can emphasize early rumor-spreading messages.

## Project Structure

```
├── data/
│   └── sample_graphs.json            # Mini rumor dataset
├── src/
│   ├── data.py                       # Dataset & feature engineering helpers
│   ├── infer.py                      # CLI inference with interpretability
│   ├── model.py                      # Temporal-aware GCN model
│   └── train.py                      # Training, cross-validation, model export
├── demo/
│   └── demo_steps.md                 # Step-by-step VS Code walkthrough
├── presentation/
│   ├── presentation_outline.md       # Slide-by-slide script for your PPT
│   └── figures/                      # (Create during demo) plots/screenshots
├── artifacts/                        # Saved checkpoints & exported assets
├── requirements.txt
└── README.md
```

## Getting Started

1. **Create and activate a virtual environment (recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

2. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run cross-validated training and export a deployable model.**
   ```bash
   python -m src.train --epochs 60 --batch-size 2 --save-model artifacts/tir_gcn.pt
   ```

   The command prints fold-level metrics, summarizes accuracy/precision/recall, retrains on the full dataset, and saves both the model weights and the fitted TF-IDF vectorizer (`artifacts/tir_gcn.vectorizer.pkl`).

4. **Run inference on any conversation graph.**
   ```bash
   python -m src.infer \
     --model artifacts/tir_gcn.pt \
     --vectorizer artifacts/tir_gcn.vectorizer.pkl \
     --data data/sample_graphs.json \
     --graph-id pheme_001
   ```

   To classify a custom conversation stored in `my_graph.json`, replace `--graph-id` with `--graph-json my_graph.json`.

## Key Ideas to Present

- **Temporal Influence Re-weighting (Novelty):** Each message pair's contribution to the GCN is amplified when their timestamps are close, modelling rapid rumor cascades.
- **Hybrid Feature Stack:** Node embeddings concatenate TF-IDF text vectors with structural (degree, betweenness) and temporal (normalized delay) signals.
- **Explainable Output:** Inference highlights the most influential posts by norm of the hidden representation, helping analysts justify predictions.

## Next Steps

- Enrich the dataset with more events (e.g., PHEME, Twitter15/16) for stronger generalization.
- Experiment with graph attention or hierarchical pooling layers.
- Deploy the inference pipeline as a REST API or Streamlit dashboard for live demonstrations.

Refer to `demo/demo_steps.md` for a full VS Code walkthrough and `presentation/presentation_outline.md` for ready-to-convert PPT talking points.
