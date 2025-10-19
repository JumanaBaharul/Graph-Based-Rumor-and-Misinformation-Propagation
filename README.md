# Graph-Based Rumor and Misinformation Propagation

This repository contains a full pipeline for analysing how rumors propagate on social media graphs using temporal-aware graph neural networks. The project includes:

- A curated mini-dataset of conversation graphs spanning **PHEME**, **Twitter15**, and **Twitter16** style events (`data/sample_graphs.json`).
- Deep learning training scripts for cross-validated evaluation and model export across multiple architectures.
- Inference utilities and a Streamlit dashboard that highlight influential posts.
- Presentation and demo assets to help you showcase the novelty of the project.

The baseline approach introduces **Temporal Influence Re-weighted GCN (TIR-GCN)**, which augments graph convolutions with propagation speed awareness so the model can emphasize early rumor-spreading messages. You can also experiment with a temporal graph attention network and a hierarchical pooling variant to compare inductive biases.

## Project Structure

```
├── data/
│   └── sample_graphs.json            # Mini rumor dataset
├── src/
│   ├── data.py                       # Dataset & feature engineering helpers
│   ├── infer.py                      # CLI inference with interpretability
│   ├── model.py                      # Temporal GCN, GAT, and hierarchical pooling models
│   └── train.py                      # Training, cross-validation, model export
├── app/
│   └── streamlit_app.py             # Streamlit dashboard for live demos
├── demo/
│   └── demo_steps.md                 # Step-by-step VS Code walkthrough
├── presentation/
│   ├── presentation_outline.md       # Slide-by-slide script for your PPT
│   └── figures/                      # (Create during demo) plots/screenshots
├── artifacts/                        # Saved checkpoints & exported assets
├── requirements.txt
└── README.md
```

## Getting Started (VS Code Workflow)

1. **Open the project folder in VS Code.** Use *File → Open Folder…* and point to the repository root. When prompted, trust the authors and allow the workspace to use the recommended settings.

2. **Create and select a virtual environment from the VS Code terminal.**
   - Launch the integrated terminal (*Terminal → New Terminal*). It opens in the project root by default.
   - Create the environment and activate it in the terminal:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
     ```
   - Once activated, press `Ctrl+Shift+P`, run **Python: Select Interpreter**, and choose the `.venv` interpreter so that VS Code uses the same environment for running and debugging.

3. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run cross-validated training and export a deployable model from the integrated terminal.** Choose an architecture:

   - Temporal GCN (default):
     ```bash
     python -m src.train --epochs 60 --batch-size 2 --save-model artifacts/tir_gcn.pt
     ```

   - Temporal Graph Attention Network:
     ```bash
     python -m src.train --architecture temporal_gat --heads 4 \
       --epochs 60 --batch-size 2 --save-model artifacts/tir_gat.pt
     ```

   - Hierarchical pooling network:
     ```bash
     python -m src.train --architecture hierarchical_pool \
       --epochs 60 --batch-size 2 --save-model artifacts/tir_hier.pt
     ```

   Each command prints fold-level metrics, summarizes accuracy/precision/recall, retrains on the full dataset, and saves both the model weights and the fitted TF-IDF vectorizer (e.g., `artifacts/tir_gcn.vectorizer.pkl`).

5. **Run inference on any conversation graph.**
   ```bash
   python -m src.infer \
     --model artifacts/tir_gcn.pt \
     --vectorizer artifacts/tir_gcn.vectorizer.pkl \
     --data data/sample_graphs.json \
     --graph-id pheme_001
   ```

   To classify a custom conversation stored in `my_graph.json`, replace `--graph-id` with `--graph-json my_graph.json`.

6. **Launch the Streamlit dashboard directly from VS Code for your demo.** Keep the virtual environment active in the integrated terminal and run:

   ```bash
   streamlit run app/streamlit_app.py
   ```

   Streamlit prints a local URL (typically `http://localhost:8501`). Hold `Ctrl` (or `Cmd` on macOS) and click the link in the terminal to open it in your browser. Use the sidebar to load the exported checkpoint/vectorizer and explore curated samples, uploaded JSON graphs, or manually entered cascades during the presentation.

## Key Ideas to Present

- **Temporal Influence Re-weighting (Novelty):** Each message pair's contribution to the GCN is amplified when their timestamps are close, modelling rapid rumor cascades.
- **Hybrid Feature Stack:** Node embeddings concatenate TF-IDF text vectors with structural (degree, betweenness) and temporal (normalized delay) signals.
- **Architectural Variants:** Compare the temporal GCN baseline with a temporal GAT (attention-driven edge weighting) and a hierarchical pooling network (multi-resolution graph summarisation).
- **Explainable Output:** Inference utilities highlight the most influential posts by norm of the hidden representation, helping analysts justify predictions.

## Next Steps

- Integrate transformer-based language encoders (e.g., BERT) for richer semantic features.
- Evaluate on full-size benchmark datasets (PHEME splits, Twitter15/16) to report comparable baselines.
- Containerise the Streamlit dashboard or wrap the inference code in a REST API for cloud deployment.

Refer to `demo/demo_steps.md` for a full VS Code walkthrough and `presentation/presentation_outline.md` for ready-to-convert PPT talking points.
