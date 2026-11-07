
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

TOP_K = 10
RRF_K = 60

PRETRAINED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_MODEL = "./trained_model"


# Natural-language search queries for the 20 categories.
# The category labels are used as proxy relevance judgments.

CATEGORY_QUERIES = {
    "alt.atheism": "discussion about atheism religion and belief",
    "comp.graphics": "computer graphics image processing and rendering",
    "comp.os.ms-windows.misc": "Microsoft Windows operating system software",
    "comp.sys.ibm.pc.hardware": "IBM PC computer hardware components",
    "comp.sys.mac.hardware": "Apple Macintosh computer hardware",
    "comp.windows.x": "X Window System Unix graphical interface",
    "misc.forsale": "items for sale buying and selling products",
    "rec.autos": "automobiles cars engines and vehicles",
    "rec.motorcycles": "motorcycles riding and maintenance",
    "rec.sport.baseball": "baseball games teams and players",
    "rec.sport.hockey": "ice hockey teams games and players",
    "sci.crypt": "cryptography encryption and computer security",
    "sci.electronics": "electronics circuits and components",
    "sci.med": "medicine health diseases and medical research",
    "sci.space": "space astronomy NASA spacecraft and shuttle",
    "soc.religion.christian": "Christianity Christian religion and church",
    "talk.politics.guns": "gun control firearms and politics",
    "talk.politics.mideast": "Middle East politics Israel and Arab conflict",
    "talk.politics.misc": "politics government and elections",
    "talk.religion.misc": "religion religious beliefs and discussions"
}


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

print("\nLoading 20 Newsgroups TEST dataset...")

data = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes")
)

documents = [
    re.sub(r"\s+", " ", text).strip()
    for text in data.data
]

labels = np.array(data.target)
categories = data.target_names

print("Total test documents:", len(documents))
print("Total categories:", len(categories))


# ==========================================
# 3. BUILD TF-IDF INDEX
# ==========================================

print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.95,
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(documents)


# ==========================================
# 4. BUILD BM25 INDEX
# ==========================================

print("Building BM25 index...")

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)


# ==========================================
# 5. LOAD BOTH SENTENCE TRANSFORMERS
# ==========================================

print("\nLoading pretrained MiniLM...")

pretrained_model = SentenceTransformer(PRETRAINED_MODEL)

print("Loading fine-tuned MiniLM...")

finetuned_model = SentenceTransformer(FINETUNED_MODEL)


# ==========================================
# 6. GENERATE DOCUMENT EMBEDDINGS
# ==========================================

print("\nGenerating pretrained embeddings...")

pretrained_embeddings = pretrained_model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nGenerating fine-tuned embeddings...")

finetuned_embeddings = finetuned_model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ==========================================
# 7. EVALUATION METRICS
# ==========================================

def calculate_metrics(ranked_ids, relevant, k=10):

    total_relevant = np.sum(relevant)

    if total_relevant == 0:
        return 0.0, 0.0, 0.0, 0.0

    ranked_relevance = relevant[ranked_ids]

    # Precision@K
    hits_at_k = ranked_relevance[:k].sum()
    precision = hits_at_k / k

    # Recall@K
    recall = hits_at_k / total_relevant

    # Average Precision (full ranking)
    cumulative_hits = np.cumsum(ranked_relevance)
    positions = np.arange(1, len(ranked_relevance) + 1)

    precision_at_positions = cumulative_hits / positions

    average_precision = (
        precision_at_positions * ranked_relevance
    ).sum() / total_relevant

    # nDCG@K
    gains = ranked_relevance[:k]
    discounts = np.log2(np.arange(2, k + 2))

    dcg = np.sum(gains / discounts)

    ideal_hits = min(total_relevant, k)
    ideal_gains = np.ones(ideal_hits)

    ideal_discounts = np.log2(
        np.arange(2, ideal_hits + 2)
    )

    idcg = np.sum(ideal_gains / ideal_discounts)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, average_precision, ndcg


# ==========================================
# 8. PRECISION-RECALL CURVE DATA
# ==========================================

def get_precision_recall_curve(scores, relevant):

    order = np.argsort(-scores)
    ranked_relevance = relevant[order]

    total_relevant = np.sum(relevant)

    cumulative_hits = np.cumsum(ranked_relevance)
    positions = np.arange(1, len(ranked_relevance) + 1)

    precision_values = cumulative_hits / positions
    recall_values = cumulative_hits / total_relevant

    # Include starting point
    precision_values = np.concatenate(([1.0], precision_values))
    recall_values = np.concatenate(([0.0], recall_values))

    return recall_values, precision_values


# ==========================================
# 9. RUN ALL RETRIEVAL METHODS
# ==========================================

method_names = [
    "TF-IDF",
    "BM25",
    "Pretrained MiniLM",
    "Fine-tuned MiniLM",
    "Hybrid RRF"
]

all_metrics = {name: [] for name in method_names}

recall_grid = np.linspace(0, 1, 101)

pr_curves = {
    name: [] for name in method_names
}

print("\nStarting evaluation...")

for category_name, query in CATEGORY_QUERIES.items():

    print("\nEvaluating:", category_name)
    print("Query:", query)

    category_id = categories.index(category_name)

    # Documents belonging to the target category are relevant
    relevant = (labels == category_id)

    # TF-IDF scores
    query_vector = vectorizer.transform([query])

    tfidf_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # BM25 scores
    bm25_scores = np.array(
        bm25.get_scores(query.lower().split())
    )

    # Pretrained MiniLM scores
    query_pretrained = pretrained_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    pretrained_scores = (
        pretrained_embeddings @ query_pretrained
    )

    # Fine-tuned MiniLM scores
    query_finetuned = finetuned_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    finetuned_scores = (
        finetuned_embeddings @ query_finetuned
    )

    # Hybrid Reciprocal Rank Fusion
    rrf_scores = np.zeros(len(documents))

    for score_array in [
        tfidf_scores,
        bm25_scores,
        finetuned_scores
    ]:

        ranking = np.argsort(-score_array)

        for rank, doc_id in enumerate(ranking, start=1):
            rrf_scores[doc_id] += 1 / (RRF_K + rank)

    score_dictionary = {
        "TF-IDF": tfidf_scores,
        "BM25": bm25_scores,
        "Pretrained MiniLM": pretrained_scores,
        "Fine-tuned MiniLM": finetuned_scores,
        "Hybrid RRF": rrf_scores
    }

    # Evaluate each method
    for method_name, scores in score_dictionary.items():

        ranked_ids = np.argsort(-scores)

        metrics = calculate_metrics(
            ranked_ids,
            relevant,
            TOP_K
        )

        all_metrics[method_name].append(metrics)

        recall_values, precision_values = (
            get_precision_recall_curve(scores, relevant)
        )

        interpolated_precision = np.interp(
            recall_grid,
            recall_values,
            precision_values
        )

        pr_curves[method_name].append(
            interpolated_precision
        )


# ==========================================
# 10. CREATE COMPARISON TABLE
# ==========================================

results = []

for method_name in method_names:

    method_results = np.array(
        all_metrics[method_name]
    )

    average_metrics = method_results.mean(axis=0)

    results.append({
        "Method": method_name,
        "Precision@10": average_metrics[0],
        "Recall@10": average_metrics[1],
        "MAP": average_metrics[2],
        "nDCG@10": average_metrics[3]
    })

results_df = pd.DataFrame(results)

print("\n========================================")
print("FINAL RETRIEVAL PERFORMANCE COMPARISON")
print("========================================")

print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    "evaluation_results.csv",
    index=False
)

print("\nSaved evaluation_results.csv")


# ==========================================
# 11. PERFORMANCE COMPARISON GRAPH
# ==========================================

metrics_to_plot = [
    "Precision@10",
    "Recall@10",
    "MAP",
    "nDCG@10"
]

for metric in metrics_to_plot:

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["Method"],
        results_df[metric]
    )

    plt.title(metric + " Comparison")
    plt.ylabel("Score")
    plt.xlabel("Retrieval Method")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.tight_layout()

    filename = (
        metric.lower().replace("@", "_")
        .replace("/", "_") + "_comparison.png"
    )

    plt.savefig(filename, dpi=300)
    plt.close()

    print("Saved:", filename)


# ==========================================
# 12. PRECISION-RECALL GRAPH
# ==========================================

plt.figure(figsize=(10, 7))

for method_name in method_names:

    mean_precision = np.mean(
        pr_curves[method_name],
        axis=0
    )

    plt.plot(
        recall_grid,
        mean_precision,
        label=method_name
    )

plt.title("Macro-Averaged Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "precision_recall_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved precision_recall_comparison.png")

print("\nEVALUATION COMPLETED SUCCESSFULLY!")
