import json
import time
from datetime import datetime
from pathlib import Path

from src.consumer_rag.chunking import load_docs, chunk_doc
from src.consumer_rag.retrieval import build_index, search
from src.consumer_rag.reranking import Reranker, fused_search
from src.consumer_rag.hybrid import BM25Retriever
from src.consumer_rag.hybrid_retrieval import hybrid_search
from src.consumer_rag.evaluate import summarize


QUESTIONS_PATH = Path("eval/holdout_questions.jsonl")
RESULTS_DIR = Path("eval/results")


def load_holdout_questions():
    questions = []

    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def load_chunks():
    docs = load_docs(Path("data/raw"))
    chunks = []

    for doc in docs:
        chunks.extend(
            chunk_doc(
                doc,
                max_chars=800,
                with_header=True,
            )
        )

    return chunks

def first_relevant_chunk_rank(results, relevant_ids):
    """
    Return the 1-based rank of the first relevant chunk.

    Returns None if no relevant chunk was retrieved.
    """
    relevant_set = set(relevant_ids)

    for rank, result in enumerate(results, start=1):
        if result["id"] in relevant_set:
            return rank

    return None


def evaluate_strategy(name, questions, retrieve_fn):
    ranks = []
    latencies = []
    question_results = []

    print(f"\n{'=' * 65}")
    print(f"STRATEGY: {name.upper()}")
    print("=" * 65)

    for index, question in enumerate(questions, start=1):
        start = time.perf_counter()

        results = retrieve_fn(question["q"])

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        rank = first_relevant_chunk_rank(
        results,
        question["relevant"],
        )

        ranks.append(rank)
        latencies.append(latency_ms)

        retrieved_ids = [
            result["id"]
            for result in results
        ]

        question_results.append({
            "question_id": index,
            "question": question["q"],
            "type": question["type"],
            "relevant": question["relevant"],
            "retrieved": retrieved_ids,
            "first_hit_rank": rank,
            "latency_ms": round(latency_ms, 2),
        })

        rank_display = rank if rank is not None else "MISS"

        print(
            f"Q{index:02d} "
            f"[{question['type']:<10}] "
            f"Rank: {str(rank_display):<5} "
            f"Time: {latency_ms:.0f} ms"
        )

    metrics = summarize(ranks)

    metrics["avg_latency_ms"] = round(
        sum(latencies) / len(latencies),
        2,
    )

    print("\nMETRICS")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return {
        "metrics": metrics,
        "questions": question_results,
    }


def main():
    questions = load_holdout_questions()

    print("HOLDOUT RETRIEVAL EVALUATION")
    print("=" * 65)
    print("Questions:", len(questions))

    # Build the shared dense index
    print("\nBuilding index...")

    col, embedder, n_chunks = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    print("Indexed chunks:", n_chunks)

    # Initialize reranker and BM25
    print("\nLoading reranker and BM25...")

    reranker = Reranker()
    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)

    strategies = {
        "dense": lambda q: search(
            col,
            embedder,
            q,
            k=5,
        ),

        "fusion": lambda q: fused_search(
            col,
            embedder,
            reranker,
            q,
            k=5,
            candidate_k=20,
        ),

        "hybrid": lambda q: hybrid_search(
            q,
            col,
            embedder,
            bm25,
            reranker,
            dense_k=5,
            bm25_k=5,
            final_k=5,
        ),
    }

    evaluation = {
        "config": {
            "embedding_model": "e5-small",
            "chunk_size": 800,
            "questions": len(questions),
            "final_k": 5,
            "fusion_candidate_k": 20,
            "hybrid_dense_k": 5,
            "hybrid_bm25_k": 5,
        },
        "strategies": {},
    }

    for name, retrieve_fn in strategies.items():
        evaluation["strategies"][name] = evaluate_strategy(
            name,
            questions,
            retrieve_fn,
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        RESULTS_DIR / f"holdout_{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(
            evaluation,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for name, result in evaluation["strategies"].items():
        metrics = result["metrics"]

        print(
            f"{name:<8} | "
            f"Hit@1: {metrics['recall@1']:.3f} | "
            f"Hit@3: {metrics['recall@3']:.3f} | "
            f"Hit@5: {metrics['recall@5']:.3f} | "
            f"MRR: {metrics['mrr']:.3f} | "
            f"Avg: {metrics['avg_latency_ms']:.0f} ms"
        )

    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    main()
