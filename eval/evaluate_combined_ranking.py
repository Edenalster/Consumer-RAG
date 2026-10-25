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


def first_relevant_rank(results, relevant_ids):
    for rank, chunk in enumerate(results, start=1):
        if chunk["id"] in relevant_ids:
            return rank
    return None


questions = load_jsonl("eval/unseen_questions.jsonl")
ground_truth = load_jsonl("eval/unseen_ground_truth.jsonl")

relevant_by_id = {
    item["id"]: set(item["relevant"])
    for item in ground_truth
}

strategies = {
    "baseline": [],
    "category_boost_10": [],
    "expanded_reranking": [],
    "combined_boost_10": [],
}


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

    category = classify_legal_query(question)

    if category == "לא ידוע" or category in question:
        expanded_question = question
    else:
        expanded_question = f"{question} {category}"

    dense_results = search(
        col,
        embedder,
        expanded_question,
        k=20,
    )

    original_reranked = reranker.rerank(
        question,
        dense_results,
    )

    expanded_reranked = reranker.rerank(
        expanded_question,
        dense_results,
    )

    original_fusion = reciprocal_rank_fusion(
        dense_results,
        original_reranked,
        dense_weight=0.4,
        sparse_weight=0.6,
    )

    expanded_fusion = reciprocal_rank_fusion(
        dense_results,
        expanded_reranked,
        dense_weight=0.4,
        sparse_weight=0.6,
    )

    variants = {
        "baseline": original_fusion[:5],

        "category_boost_10": boost_by_category_linear(
            original_fusion,
            category,
            boost=10,
        )[:5],

        "expanded_reranking": expanded_fusion[:5],

        "combined_boost_10": boost_by_category_linear(
            expanded_fusion,
            category,
            boost=10,
        )[:5],
    }

    relevant_ids = relevant_by_id[question_id]

    print("\n" + "=" * 65)
    print(f"QUESTION: {question_id}")
    print(f"CATEGORY: {category}")
    print("=" * 65)

    for name, results in variants.items():
        rank = first_relevant_rank(
            results,
            relevant_ids,
        )

        strategies[name].append(rank)

        print(
            f"{name:22} "
            f"-> first relevant rank: {rank}"
        )


print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)

for name, ranks in strategies.items():
    metrics = summarize(ranks)

    print(f"\n{name}")

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")