# Presentation Outline: Temporal Influence Re-weighted GCN for Rumor Propagation

## Slide 1 – Title
- **Project Title:** Temporal Influence Re-weighted GCN for Rumor & Misinformation Propagation
- Team members & course details
- Eye-catching visualization of a conversation graph (add during rehearsal)

## Slide 2 – Objective
- Detect and characterize rumor cascades using graph-based deep learning.
- Quantify how temporal dynamics impact misinformation spread.
- Provide explainable insights for analysts and fact-checkers.

## Slide 3 – Research Gap & Motivation
- Traditional classifiers treat posts independently → ignore propagation structure.
- Existing graph models rarely incorporate **time-sensitive influence**.
- Public demand for rapid, explainable rumor detection in crisis scenarios.

## Slide 4 – Proposed Methodology
1. Curate conversation graphs from verified & rumor events (sampled from PHEME-like narratives).
2. Engineer hybrid node features:
   - TF-IDF textual embeddings
   - Structural signals (degree, betweenness)
   - Temporal lag from source post
3. **Temporal Influence Re-weighted GCN (Novelty):**
   - Dynamically boosts edges with small time gaps.
   - Captures fast rumor amplification compared to verified news.
4. Cross-validated training and model export for deployment.

## Slide 5 – Architecture Diagram
- Dataset ingestion → Feature engineering → Temporal GCN layers → Graph pooling → Binary classifier.
- Highlight re-weighted adjacency formula: \(\tilde{A}_{ij} = A_{ij} (1 + \alpha e^{-\beta |t_i - t_j|})\).
- Mention PyTorch-based implementation.

## Slide 6 – Experimental Setup
- Dataset: 6 curated conversation graphs (3 rumor, 3 verified).
- Training: 3-fold stratified CV, 60 epochs, Adam optimizer.
- Evaluation metrics: Accuracy, Precision/Recall/F1 per class.
- Hardware: CPU-friendly (runs in under a minute).

## Slide 7 – Results & Findings
- Table of cross-validation metrics (populate from training logs).
- Confusion matrices per fold (optional screenshot from terminal).
- Insight: Temporal re-weighting improves recall on rumor class vs. vanilla GCN baseline (include comparison if available).

## Slide 8 – Qualitative Analysis
- Show inference output highlighting top contributing posts.
- Discuss how early, emotional messages receive higher influence scores.
- Link to ethical considerations (transparency, human-in-the-loop verification).

## Slide 9 – Conclusion
- Summarize key outcomes: accurate, interpretable rumor detection using temporal graphs.
- Novelty reiterated: temporal influence module + explainable node ranking.

## Slide 10 – Future Scope
- Scale to larger datasets (Twitter15/16, Weibo) & multilingual embeddings.
- Integrate transformer-based text encoders (e.g., BERT) for richer semantics.
- Deploy as dashboard/API for newsroom or public policy stakeholders.

## Slide 11 – Live Demo Checklist
- Launch VS Code environment.
- Run training script and show metrics.
- Execute inference on rumor & verified graphs.
- Display highlighted top contributing posts.
- Optional: Visualize graph using NetworkX/Matplotlib.

## Slide 12 – Q&A
- Include references to seminal papers on rumor detection and GNNs.
- Invite questions about scalability, ethics, or deployment.

> **Tip:** Export this outline to PowerPoint (e.g., via Microsoft PowerPoint or Google Slides) and replace bullet lists with visuals, tables, and charts generated from the provided code outputs.
