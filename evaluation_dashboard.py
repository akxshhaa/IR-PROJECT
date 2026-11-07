
import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="IR Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

st.title("20 Newsgroups - Evaluation Dashboard")

st.write(
    "Comparative evaluation of five Information Retrieval "
    "approaches using the 20 Newsgroups dataset."
)

st.markdown("---")


# ==========================================
# LOAD RESULTS
# ==========================================

results_path = BASE_DIR / "evaluation_results.csv"

if not results_path.exists():
    st.error("evaluation_results.csv not found. Run evaluate.py first.")
    st.stop()

df = pd.read_csv(results_path)

required_columns = [
    "Method",
    "Precision@10",
    "Recall@10",
    "MAP",
    "nDCG@10"
]

if not all(col in df.columns for col in required_columns):
    st.error("Evaluation results file has an unexpected format.")
    st.stop()


# ==========================================
# TOP PERFORMANCE METRICS
# ==========================================

st.header("Overall Performance")

best_precision = df.loc[df["Precision@10"].idxmax()]
best_map = df.loc[df["MAP"].idxmax()]
best_ndcg = df.loc[df["nDCG@10"].idxmax()]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Best Precision@10",
    f"{best_precision['Precision@10']:.3f}",
    best_precision["Method"]
)

col2.metric(
    "Best MAP",
    f"{best_map['MAP']:.3f}",
    best_map["Method"]
)

col3.metric(
    "Best nDCG@10",
    f"{best_ndcg['nDCG@10']:.3f}",
    best_ndcg["Method"]
)

st.markdown("---")


# ==========================================
# COMPARISON TABLE
# ==========================================

st.header("Retrieval Method Comparison")

display_df = df.copy()

for col in required_columns[1:]:
    display_df[col] = display_df[col].round(4)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# METRIC COMPARISON CHARTS
# ==========================================

st.header("Performance Comparison")

metric_tabs = st.tabs([
    "Precision@10",
    "Recall@10",
    "MAP",
    "nDCG@10"
])

chart_files = {
    "Precision@10": "precision_10_comparison.png",
    "Recall@10": "recall_10_comparison.png",
    "MAP": "map_comparison.png",
    "nDCG@10": "ndcg_10_comparison.png"
}

for tab, metric in zip(
    metric_tabs,
    chart_files.keys()
):

    with tab:

        chart_path = BASE_DIR / chart_files[metric]

        if chart_path.exists():
            st.image(
                str(chart_path),
                use_container_width=True
            )
        else:
            st.warning(f"{chart_files[metric]} not found.")


# ==========================================
# PRECISION-RECALL CURVE
# ==========================================

st.markdown("---")

st.header("Precision-Recall Analysis")

pr_path = BASE_DIR / "precision_recall_comparison.png"

if pr_path.exists():

    st.image(
        str(pr_path),
        use_container_width=True
    )

    st.caption(
        "Macro-averaged Precision-Recall curves across "
        "the 20 category queries."
    )

else:
    st.warning("Precision-Recall graph not found.")


# ==========================================
# PRETRAINED VS FINE-TUNED COMPARISON
# ==========================================

st.markdown("---")

st.header("Impact of Model Fine-Tuning")

pretrained = df[
    df["Method"] == "Pretrained MiniLM"
]

finetuned = df[
    df["Method"] == "Fine-tuned MiniLM"
]

if not pretrained.empty and not finetuned.empty:

    p = pretrained.iloc[0]
    f = finetuned.iloc[0]

    metrics = [
        "Precision@10",
        "Recall@10",
        "MAP",
        "nDCG@10"
    ]

    comparison = []

    for metric in metrics:

        old_value = p[metric]
        new_value = f[metric]

        relative_change = (
            ((new_value - old_value) / old_value) * 100
            if old_value != 0 else 0
        )

        comparison.append({
            "Metric": metric,
            "Pretrained": round(old_value, 4),
            "Fine-tuned": round(new_value, 4),
            "Relative improvement (%)": round(
                relative_change, 2
            )
        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        comparison_df.set_index("Metric")[
            ["Pretrained", "Fine-tuned"]
        ]
    )

else:
    st.warning("Pretrained or fine-tuned model results are missing.")


# ==========================================
# EXPERIMENTAL FINDINGS
# ==========================================

st.markdown("---")

st.header("Key Experimental Findings")

if not pretrained.empty and not finetuned.empty:

    precision_gain = (
        f["Precision@10"] - p["Precision@10"]
    ) * 100

    map_gain = (
        (f["MAP"] - p["MAP"]) / p["MAP"] * 100
        if p["MAP"] != 0 else 0
    )

    ndcg_gain = (
        (f["nDCG@10"] - p["nDCG@10"]) / p["nDCG@10"] * 100
        if p["nDCG@10"] != 0 else 0
    )

    st.markdown(
        f"""
        - Fine-tuned MiniLM achieved
          **{f['Precision@10']:.3f} Precision@10**.
        - Precision@10 increased by
          **{precision_gain:.2f} percentage points**
          compared with pretrained MiniLM.
        - MAP improved by **{map_gain:.2f}%**.
        - nDCG@10 improved by **{ndcg_gain:.2f}%**.
        - Fine-tuning improved the ranking performance
          under the category-based evaluation setup.
        """
    )

st.info(
    "Evaluation note: Relevance is defined using the "
    "20 Newsgroups category labels. Each category has one "
    "representative search query. These results therefore "
    "measure category-based retrieval performance rather "
    "than human-judged relevance."
)

st.markdown("---")

st.caption(
    "Information Retrieval Project | 20 Newsgroups Dataset"
)
