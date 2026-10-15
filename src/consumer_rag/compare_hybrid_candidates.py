import json
import time
from pathlib import Path

from src.consumer_rag.chunking import load_docs, chunk_doc
from src.consumer_rag.retrieval import build_index
from src.consumer_rag.reranking import Reranker
from src.consumer_rag.hybrid import BM25Retriever
from src.consumer_rag.hybrid_retrieval import hybrid_search
from src.consumer_rag.evaluate import summarize


QUESTIONS_PATH = Path("eval/holdout_questions.jsonl")

CONFIGS = [
    (5, 5),
    (10, 5),
    (15, 5),
]


def first_relevant_rank(results, relevant_ids):
    relevant = set(relevant_ids)

    for rank, result in enumerate(results, start=1):
        if result["id"] in relevant:
            return rank

    return None


def main():
    questions = [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print("Building index...")

    col, embedder, n_chunks = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

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

    bm25 = BM25Retriever(chunks)
    reranker = Reranker()

    print("Indexed chunks:", n_chunks)
    print("Questions:", len(questions))

    all_results = {}

    for dense_k, bm25_k in CONFIGS:
        name = f"hybrid_{dense_k}_{bm25_k}"

        print("\n" + "=" * 65)
        print("CONFIG:", name)
        print("=" * 65)

        ranks = []
        latencies = []
        details = []

        for index, question in enumerate(
            questions, start=1
        ):
            start = time.perf_counter()

            results = hybrid_search(
                question["q"],
                col,
                embedder,
                bm25,
                reranker,
                dense_k=dense_k,
                bm25_k=bm25_k,
                final_k=5,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            rank = first_relevant_rank(
                results,
                question["relevant"],
            )

            ranks.append(rank)
            latencies.append(elapsed_ms)

            details.append({
                "question_id": index,
                "type": question["type"],
                "rank": rank,
                "retrieved": [
                    result["id"]
                    for result in results
                ],
                "latency_ms": round(elapsed_ms, 2),
            })

            print(
                f"Q{index:02d}: "
                f"Rank={rank if rank is not None else 'MISS'} "
                f"Time={elapsed_ms:.0f}ms"
            )

        metrics = summarize(ranks)
        metrics["avg_latency_ms"] = round(
            sum(latencies) / len(latencies),
            2,
        )

        all_results[name] = {
            "config": {
                "dense_k": dense_k,
                "bm25_k": bm25_k,
                "final_k": 5,
            },
            "metrics": metrics,
            "questions": details,
        }

    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    for name, result in all_results.items():
        m = result["metrics"]

        print(
            f"{name:<15} | "
            f"Hit@1={m['recall@1']:.3f} | "
            f"Hit@3={m['recall@3']:.3f} | "
            f"Hit@5={m['recall@5']:.3f} | "
            f"MRR={m['mrr']:.3f} | "
            f"Avg={m['avg_latency_ms']:.0f}ms"
        )

    output_path = Path(
        "eval/results/hybrid_candidate_comparison.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
