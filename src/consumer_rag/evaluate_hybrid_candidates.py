from pathlib import Path
from statistics import mean
from time import perf_counter

from .chunking import load_docs, chunk_doc
from .hybrid import BM25Retriever
from .retrieval import ROOT, build_index, search
from .reranking import Reranker, fused_search
from .evaluate import load_questions, first_hit_rank, summarize


def build_bm25():
    docs = load_docs(ROOT / "data" / "raw")

    chunks = []

    for doc in docs:
        chunks.extend(
            chunk_doc(
                doc,
                max_chars=800,
                with_header=True,
            )
        )

    return BM25Retriever(chunks)


def hybrid_search(
    question,
    col,
    embedder,
    bm25,
    reranker,
    dense_k=5,
    bm25_k=5,
    final_k=5,
):
    dense_results = search(
        col,
        embedder,
        question,
        k=dense_k,
    )

    bm25_results = bm25.search(
        question,
        k=bm25_k,
    )

    seen = set()
    candidates = []

    for result in dense_results + bm25_results:
        if result["id"] not in seen:
            seen.add(result["id"])
            candidates.append(result)

    reranked = reranker.rerank(
        question,
        candidates,
    )

    return reranked[:final_k]


def main():
    questions = load_questions(
        str(ROOT / "eval" / "questions.jsonl")
    )

    print("Loading models and retrievers...")

    col, embedder, _ = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    bm25 = build_bm25()
    reranker = Reranker()

    strategies = {
        "fusion_20": [],
        "hybrid_5_5": [],
    }

    latencies = {
        "fusion_20": [],
        "hybrid_5_5": [],
    }

    print("\nEvaluating retrieval strategies...")

    for question in questions:
        query = question["q"]

        # Existing Fusion baseline
        start = perf_counter()

        fusion_results = fused_search(
            col,
            embedder,
            reranker,
            query,
            k=5,
            candidate_k=20,
        )

        latencies["fusion_20"].append(
            (perf_counter() - start) * 1000
        )

        # New Hybrid Candidate Retrieval
        start = perf_counter()

        hybrid_results = hybrid_search(
            query,
            col,
            embedder,
            bm25,
            reranker,
            dense_k=5,
            bm25_k=5,
            final_k=5,
        )

        latencies["hybrid_5_5"].append(
            (perf_counter() - start) * 1000
        )

        for name, results in [
            ("fusion_20", fusion_results),
            ("hybrid_5_5", hybrid_results),
        ]:
            rank = first_hit_rank(
                results,
                doc=question.get("doc"),
                phrase=question.get("phrase"),
                relevant=question.get("relevant"),
            )

            strategies[name].append(rank)

    print("\nHYBRID RETRIEVAL EVALUATION")
    print("=" * 90)

    print(
        f"{'Strategy':<18}"
        f"{'Hit@1':<12}"
        f"{'Hit@3':<12}"
        f"{'Hit@5':<12}"
        f"{'MRR':<12}"
        f"{'Mean Latency':<15}"
    )

    print("-" * 90)

    for name, ranks in strategies.items():
        metrics = summarize(ranks)

        print(
            f"{name:<18}"
            f"{metrics['recall@1']:<12.3f}"
            f"{metrics['recall@3']:<12.3f}"
            f"{metrics['recall@5']:<12.3f}"
            f"{metrics['mrr']:<12.3f}"
            f"{mean(latencies[name]):<15.2f}"
        )

        print(f"\nQuestions evaluated: {len(questions)}")

    print("\nPER-QUESTION COMPARISON")
    print("=" * 75)

    print(
        f"{'QID':<8}"
        f"{'Fusion Rank':<16}"
        f"{'Hybrid Rank':<16}"
        f"{'Result':<15}"
    )

    print("-" * 75)

    for i, (fusion_rank, hybrid_rank) in enumerate(
        zip(
            strategies["fusion_20"],
            strategies["hybrid_5_5"],
        ),
        start=1,
    ):
        fusion_value = fusion_rank if fusion_rank is not None else 999
        hybrid_value = hybrid_rank if hybrid_rank is not None else 999

        if hybrid_value < fusion_value:
            result = "IMPROVED"
        elif hybrid_value > fusion_value:
            result = "WORSE"
        else:
            result = "SAME"

        print(
            f"{i:<8}"
            f"{str(fusion_rank):<16}"
            f"{str(hybrid_rank):<16}"
            f"{result:<15}"
        )


if __name__ == "__main__":
    main()
    
