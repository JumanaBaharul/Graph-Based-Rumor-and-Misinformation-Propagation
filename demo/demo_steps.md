# VS Code Demo Walkthrough

Follow these steps to deliver a smooth live demonstration of the Temporal Influence Re-weighted GCN project inside Visual Studio Code.

## 1. Environment Setup
1. Open the repository folder in VS Code (`File → Open Folder`).
2. Install the **Python** and **Jupyter** extensions if they are not already available.
3. Open the integrated terminal (`Ctrl + ``).
4. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
5. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Data Familiarisation
1. In the VS Code explorer, open `data/sample_graphs.json`.
2. Explain the structure: nodes with text/timestamps and the list of edges.
3. Highlight how rumor vs verified graphs differ in narrative tone and reply patterns.

## 3. Run Cross-Validated Training
1. Create a new terminal tab and execute (choose your preferred architecture):
   ```bash
   # Temporal GCN baseline
   python -m src.train --epochs 60 --batch-size 2 --save-model artifacts/tir_gcn.pt

   # Temporal GAT variant (optional)
   python -m src.train --architecture temporal_gat --heads 4 \
     --epochs 60 --batch-size 2 --save-model artifacts/tir_gat.pt
   ```
2. Narrate the console output:
   - Fold-wise loss progression.
   - Precision/recall/F1 scores for rumor vs verified classes.
   - Saved artifacts: `tir_gcn.pt` and `tir_gcn.vectorizer.pkl`.
3. Screenshot or copy the metric summary for the presentation slides.

## 4. Inspect Saved Artifacts
1. Refresh the explorer to show the new files in `artifacts/`.
2. Open `tir_gcn.pt` (binary) in hex viewer or simply point out its presence.
3. Emphasise reproducibility: rerunning training overwrites the checkpoint.

## 5. Interactive Inference
1. Prepare a small JSON file for a live test (optional). Example snippet:
   ```json
   {
     "id": "live_demo",
     "label": 1,
     "nodes": [
       {"id": 0, "user": "rumor_seed", "text": "Hearing the bridge collapsed!", "timestamp": "2022-01-01T09:00:00Z"},
       {"id": 1, "user": "witness", "text": "I just crossed it 5 mins ago. Nothing happened.", "timestamp": "2022-01-01T09:04:00Z"},
       {"id": 2, "user": "panic_bot", "text": "Sharing photos of the collapse now!", "timestamp": "2022-01-01T09:05:00Z"}
     ],
     "edges": [[0,1],[0,2],[1,2]]
   }
   ```
2. Run inference on a built-in sample (architecture auto-detected from the checkpoint):
   ```bash
   python -m src.infer \
     --model artifacts/tir_gcn.pt \
     --vectorizer artifacts/tir_gcn.vectorizer.pkl \
     --data data/sample_graphs.json \
     --graph-id pheme_005
   ```
3. Explain the output:
   - Probability of rumor vs verification.
   - Ranked list of most influential posts.
4. Optionally, classify the custom JSON file:
   ```bash
   python -m src.infer \
     --model artifacts/tir_gcn.pt \
     --vectorizer artifacts/tir_gcn.vectorizer.pkl \
     --data data/sample_graphs.json \
     --graph-json live_demo.json
   ```

5. Showcase the Streamlit dashboard (bonus):
   ```bash
   streamlit run app/streamlit_app.py
   ```
   - Point the sidebar inputs to the exported checkpoint/vectorizer.
   - Switch between dataset samples, uploaded JSON, or manual entry to highlight interactivity.

## 6. Visualise a Graph (Optional Bonus)
1. Create a quick plotting script or open a Jupyter notebook:
   ```python
   import json
   import networkx as nx
   import matplotlib.pyplot as plt

   with open("data/sample_graphs.json") as f:
       graphs = json.load(f)

   g = nx.Graph()
   sample = next(item for item in graphs if item["id"] == "pheme_001")
   g.add_edges_from(sample["edges"])
   nx.draw(g, with_labels=True)
   plt.show()
   ```
2. Use the VS Code interactive window to render the plot.

## 7. Wrap-Up Talking Points
- Reiterate novelty: temporal influence weighting + explainable inference.
- Mention future improvements (scaling, richer embeddings, deployment).
- Invite questions.

> **Pro tip:** Rehearse once end-to-end to capture screenshots of metrics and inference outputs for your PowerPoint slides.
