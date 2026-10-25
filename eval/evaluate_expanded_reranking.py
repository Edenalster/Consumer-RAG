import json
from pathlib import Path

from dotenv import load_dotenv

from src.consumer_rag.retrieval import build_index, search
from src.consumer_rag.reranking import Reranker
from src.consumer_rag.hybrid import reciprocal_rank_fusion
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

results_by_strategy = {
    "original_reranking": [],
    "expanded_reranking": [],
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

    relevant_ids = relevant_by_id[question_id]

    variants = {
        "original_reranking": question,
        "expanded_reranking": expanded_question,
    }

    print("\n" + "=" * 65)
    print(f"QUESTION: {question_id}")
    print(f"CATEGORY: {category}")
    print("=" * 65)

    for strategy_name, rerank_query in variants.items():
        reranked_results = reranker.rerank(
            rerank_query,
            dense_results,
        )

        fused_results = reciprocal_rank_fusion(
            dense_results,
            reranked_results,
            dense_weight=0.4,
            sparse_weight=0.6,
        )

        top_5 = fused_results[:5]

        rank = first_relevant_rank(
            top_5,
            relevant_ids,
        )

        results_by_strategy[strategy_name].append(rank)

        print(
            f"{strategy_name:20} "
            f"-> first relevant rank: {rank}"
        )


print("\n" + "=" * 65)
print("FINAL RESULTS")
print("=" * 65)

for strategy_name, ranks in results_by_strategy.items():
    metrics = summarize(ranks)

    print(f"\n{strategy_name}")

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")