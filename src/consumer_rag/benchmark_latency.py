import json
import time
from pathlib import Path
from statistics import mean, median

from .retrieval import ROOT, build_index, search
from .reranking import Reranker, fused_search
from .context_expansion import expand_context


STRATEGIES = ["dense", "fusion", "fusion_expanded"]


def load_questions(path: Path) -> list[str]:
    return [
        json.loads(line)["q"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_retrieval(strategy, question, col, embedder, reranker):
    if strategy == "dense":
        return search(col, embedder, question, k=5)

    results = fused_search(
        col,
        embedder,
        reranker,
        question,
        k=5,
        candidate_k=20,
    )

    if strategy == "fusion_expanded":
        results = expand_context(
            col=col,
            embedder=embedder,
            query=question,
            base_results=results,
            extra_k=5,
        )

    return results

def benchmark_candidate_k(questions, col, embedder, reranker):
    candidate_values = [5, 10, 20]

    print("\nRERANKER CANDIDATE BENCHMARK")
    print("=" * 55)
    print(f"{'Candidate K':<15}{'Mean (ms)':<15}{'Median (ms)':<15}")
    print("-" * 55)

    for candidate_k in candidate_values:
        times = []

        for question in questions:
            start = time.perf_counter()

            fused_search(
                col,
                embedder,
                reranker,
                question,
                k=5,
                candidate_k=candidate_k,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        print(
            f"{candidate_k:<15}"
            f"{mean(times):<15.2f}"
            f"{median(times):<15.2f}"
        )


def main():
    questions = load_questions(ROOT / "eval" / "questions.jsonl")

    print("Loading models and index...")
    col, embedder, _ = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )
    reranker = Reranker()

    results = {strategy: [] for strategy in STRATEGIES}

    print("\nRunning retrieval latency benchmark...")

    for question in questions:
        for strategy in STRATEGIES:
            start = time.perf_counter()

            run_retrieval(
                strategy,
                question,
                col,
                embedder,
                reranker,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            results[strategy].append(elapsed_ms)

    print("\nRETRIEVAL LATENCY RESULTS")
    print("=" * 60)
    print(f"{'Strategy':<20}{'Mean (ms)':<15}{'Median (ms)':<15}")
    print("-" * 60)

    for strategy, times in results.items():
        print(
            f"{strategy:<20}"
            f"{mean(times):<15.2f}"
            f"{median(times):<15.2f}"
        )

    print(f"\nQuestions benchmarked: {len(questions)}")
    benchmark_candidate_k(
    questions,
    col,
    embedder,
    reranker,
)



if __name__ == "__main__":
    main()