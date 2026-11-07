# 20 Newsgroups Information Retrieval Project

## Why this project?
This is the easiest practical option from the assignment list to get running: scikit-learn downloads the 20 Newsgroups dataset automatically. It supports the four required ranking methods and a simple interactive demo.

## Setup
1. Install Python 3.10+.
2. In this folder, run:
   ```bash
   python -m venv .venv
   ```
   Activate it:
   - Windows: `.venv\\Scripts\\activate`
   - macOS/Linux: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. On first launch, the dataset and semantic model download automatically. Internet is needed for this first run.

## Implemented
- TF-IDF retrieval
- BM25 retrieval
- Sentence Transformer semantic retrieval (`all-MiniLM-L6-v2`)
- Hybrid Reciprocal Rank Fusion (RRF), with k=60
- Search interface with top results
- Sample Precision@10, Recall@10, AP, and nDCG@10 evaluation

## Important evaluation note
20 Newsgroups provides topic labels, not query-document relevance judgments. This starter uses topic-category matching as a relevance proxy for a small evaluation set. Explain this limitation in your report; do not present it as human relevance annotation.

## Suggested report outline
1. Abstract
2. Introduction and problem statement
3. Dataset overview (20 Newsgroups)
4. Preprocessing
5. Retrieval methods (TF-IDF, BM25, dense semantic, RRF)
6. Evaluation metrics and relevance-proxy limitation
7. Results table and graph
8. Conclusion and future work
