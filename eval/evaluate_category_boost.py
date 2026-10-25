import json
from pathlib import Path

from dotenv import load_dotenv

from src.consumer_rag.retrieval import build_index, search
from src.consumer_rag.reranking import Reranker
from src.consumer_rag.hybrid import reciprocal_rank_fusion
from src.consumer_rag.category_boost import boost_by_category_linear
from src.consumer_rag.legal_classifier import classify_legal_query
from src.consumer_rag.evaluate import summarize


load_dotenv(dotenv_path=".env")


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


questions = load_jsonl("eval/unseen_questions.jsonl")
ground_truth = load_jsonl("eval/unseen_ground_truth.jsonl")

relevant_by_id = {
    item["id"]: set(item["relevant"])
    for item in ground_truth
}

strategies = {
    "baseline": [],
    "boost_5": [],
    "boost_10": [],
    "boost_15": [],
}


def first_relevant_rank(results, relevant_ids):
    for rank, chunk in enumerate(results, start=1):
        if chunk["id"] in relevant_ids:
            return rank

    return None


print("Loading retrieval models...")

col, embedder, n_chunks = build_index(
    model_key="e5-small",
    max_chars=800,
    with_header=True,
)

reranker = Reranker()


for item in questions:
    question_id = item["id"]
    question = item["q"]

    print(f"\n{'=' * 65}")
    print(f"QUESTION: {question_id}")
    print(f"{'=' * 65}")

    category = classify_legal_query(question)

    print(f"Category: {category}")

    if category == "לא ידוע":
        expanded_question = question
    elif category in question:
        expanded_question = question
    else:
        expanded_question = f"{question} {category}"

    dense_results = search(
        col,
        embedder,
        expanded_question,
        k=20,
    )

    reranked_results = reranker.rerank(
        question,
        dense_results,
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        reranked_results,
        dense_weight=0.4,
        sparse_weight=0.6,
    )

    relevant_ids = relevant_by_id[question_id]

    variants = {
        "baseline": fused_results[:5],
        "boost_5": boost_by_category_linear(
            fused_results, category, boost=5
        )[:5],
        "boost_10": boost_by_category_linear(
            fused_results, category, boost=10
        )[:5],
        "boost_15": boost_by_category_linear(
            fused_results, category, boost=15
        )[:5],
    }

    for name, results in variants.items():
        rank = first_relevant_rank(
            results,
            relevant_ids,
        )

        strategies[name].append(rank)

        print(f"{name:12} -> first relevant rank: {rank}")


print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)

for name, ranks in strategies.items():
    metrics = summarize(ranks)

    print(f"\n{name}")

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")