import streamlit as st
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict

st.set_page_config(page_title="20 Newsgroups Search Engine", page_icon="🔎", layout="wide")
st.title("🔎 20 Newsgroups Information Retrieval System")
st.caption("TF-IDF • BM25 • Sentence Transformers • Hybrid Reciprocal Rank Fusion")

@st.cache_resource(show_spinner="Loading dataset and building indexes (first run may take a few minutes)…")
def load_system():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    docs = [re.sub(r"\\s+", " ", x).strip() for x in data.data]
    labels = data.target
    names = data.target_names

    tfidf = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)
    X = tfidf.fit_transform(docs)
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    model = SentenceTransformer("./trained_model")
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
    return docs, labels, names, tfidf, X, bm25, model, embeddings

with st.spinner("Preparing search engine…"):
    docs, labels, names, tfidf, X, bm25, model, embeddings = load_system()

def ranks_from_scores(scores):
    return np.argsort(-scores)

def search(query, method, k=10):
    qv = tfidf.transform([query])
    sparse_scores = cosine_similarity(qv, X).ravel()
    bm_scores = np.asarray(bm25.get_scores(query.lower().split()))
    dense_q = model.encode([query], normalize_embeddings=True)
    dense_scores = embeddings @ dense_q[0]

    if method == "TF-IDF":
        scores = sparse_scores
    elif method == "BM25":
        scores = bm_scores
    elif method == "Semantic":
        scores = dense_scores
    else:
        # Reciprocal Rank Fusion: sum 1/(60 + rank) across three retrievers
        score_map = defaultdict(float)
        for arr in (sparse_scores, bm_scores, dense_scores):
            order = ranks_from_scores(arr)
            for rank, idx in enumerate(order, start=1):
                score_map[int(idx)] += 1.0 / (60 + rank)
        scores = np.zeros(len(docs))
        for idx, val in score_map.items():
            scores[idx] = val
    return ranks_from_scores(scores)[:k], scores

st.sidebar.header("Search settings")
method = st.sidebar.selectbox("Ranking method", ["Hybrid RRF", "TF-IDF", "BM25", "Semantic"])
top_k = st.sidebar.slider("Number of results", 5, 20, 10)
query = st.text_input("Enter your search query", placeholder="e.g. space shuttle launch")

if query.strip():
    ids, scores = search(query.strip(), method, top_k)
    st.subheader(f"Top {len(ids)} results · {method}")
    for rank, idx in enumerate(ids, 1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {names[labels[idx]]}**  ·  Score: `{scores[idx]:.4f}`")
            st.write(docs[idx][:900] + ("…" if len(docs[idx]) > 900 else ""))
else:
    st.info("Try a query such as **computer graphics**, **space shuttle**, **baseball**, or **religion**.")

st.divider()
st.subheader("Evaluation (category-based relevance proxy)")
st.caption("For this starter, documents in the query's predicted topic category are treated as relevant. This is a transparent proxy, not human relevance judgments.")
topic_queries = {
    "comp.graphics": "computer graphics image rendering",
    "rec.sport.baseball": "baseball pitcher game",
    "sci.space": "space shuttle nasa orbit",
    "talk.politics.misc": "politics government election",
    "sci.med": "medical health disease",
}
if st.button("Run sample evaluation"):
    rows=[]
    for topic, q in topic_queries.items():
        relevant_label = names.index(topic)
        ids, _ = search(q, method, 10)
        rel = [1 if labels[i] == relevant_label else 0 for i in ids]
        precision10 = sum(rel)/10
        total_relevant = int(np.sum(labels == relevant_label))
        recall10 = sum(rel)/total_relevant if total_relevant else 0
        dcg = sum(r/(np.log2(pos+1)) for pos,r in enumerate(rel, start=1))
        ideal = sum(1/(np.log2(pos+1)) for pos in range(1, min(10,total_relevant)+1))
        ndcg = dcg/ideal if ideal else 0
        ap = sum(sum(rel[:i])/i for i in range(1,11) if rel[i-1])/max(sum(rel),1)
        rows.append({"Query/topic":topic,"Precision@10":precision10,"Recall@10":recall10,"AP@10":ap,"nDCG@10":ndcg})
    st.dataframe(rows, use_container_width=True)
    st.bar_chart({r["Query/topic"]:r["nDCG@10"] for r in rows})
