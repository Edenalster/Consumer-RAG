import json
from copy import deepcopy
from datetime import datetime

from dotenv import load_dotenv

from .retrieval import ROOT, build_index
from .reranking import Reranker
from .hybrid import BM25Retriever
from .chunking import load_docs, chunk_doc
from .rag import ask
from .evaluate import load_questions


BASELINE_FILE = (
    ROOT
    / "eval"
    / "results"
    / "comparison_20261014_162307.json"
)


def main():
    load_dotenv(dotenv_path=ROOT / ".env")

    # Load existing baseline answers
    baseline = json.loads(
        BASELINE_FILE.read_text(encoding="utf-8")
    )

    # Load original questions to recover QIDs
    all_questions = load_questions(
        str(ROOT / "eval" / "questions.jsonl")
    )

    question_ids = {
        item["q"]: index
        for index, item in enumerate(all_questions, start=1)
    }

    # Work on a copy to preserve baseline data
    results = deepcopy(baseline)

    # Validate all questions before running Gemini
    expected_ids = {4, 5, 6, 8, 10, 11}

    actual_ids = {
        question_ids[item["question"]]
        for item in results
    }

    if actual_ids != expected_ids:
        raise ValueError(
            f"Unexpected questions: {actual_ids}"
        )

    required_strategies = {
        "dense",
        "fusion",
        "fusion_expanded",
    }

    for item in results:
        for strategy in required_strategies:
            result = item["strategies"].get(strategy)

            if not result or result.get("status") != "success":
                raise ValueError(
                    f"Missing successful {strategy} "
                    f"for question: {item['question']}"
                )

        item["question_id"] = question_ids[item["question"]]

    print("Baseline validated successfully!", flush=True)

    # Initialize retrieval components
    print("Building Dense index...", flush=True)

    col, embedder, _ = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    print("Loading Reranker...", flush=True)
    reranker = Reranker()

    print("Building BM25...", flush=True)

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

    bm25 = BM25Retriever(chunks)

    # Generate answers using Hybrid only
    for item in results:
        qid = item["question_id"]
        question = item["question"]

        print(
            f"\nQID {qid}: Running hybrid...",
            flush=True,
        )

        try:
            result = ask(
                question=question,
                col=col,
                embedder=embedder,
                k=5,
                strategy="hybrid",
                reranker=reranker,
                bm25=bm25,
            )

            item["strategies"]["hybrid"] = {
                "answer": result["answer"],
                "sources": result["sources"],
                "status": "success",
            }

            print("  Success!", flush=True)

        except Exception as error:
            item["strategies"]["hybrid"] = {
                "status": "error",
                "error": str(error),
            }

            print(f"  Error: {error}", flush=True)

    # Save new comparison file
    output_dir = ROOT / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        output_dir
        / f"comparison_hybrid_{timestamp}.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()