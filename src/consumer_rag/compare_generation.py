import json
from datetime import datetime

from dotenv import load_dotenv

from .retrieval import build_index, ROOT
from .reranking import Reranker
from .hybrid import BM25Retriever
from .chunking import load_docs, chunk_doc
from .rag import ask
from .evaluate import load_questions


def main():
    # Load Gemini API key
    load_dotenv(dotenv_path=ROOT / ".env")

    # Load evaluation questions
    all_questions = load_questions(
        str(ROOT / "eval" / "questions.jsonl")
    )

    # Keep the original question IDs
    QUESTION_IDS = {4, 5, 6, 8, 10, 11}

    questions = [
        (index, question)
        for index, question in enumerate(all_questions, start=1)
        if index in QUESTION_IDS
    ]

    print("Building retrieval index...", flush=True)

    col, embedder, _ = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    print("Loading reranker...", flush=True)
    reranker = Reranker()

    print("Building BM25 retriever...", flush=True)

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

    results = []

    strategies = [
        "dense",
        "fusion",
        "fusion_expanded",
        "hybrid",
    ]

    for question_id, item in questions:
        question = item["q"]

        print(
            f"\nQuestion {question_id}: {question}",
            flush=True,
        )

        comparison = {
            "question_id": question_id,
            "question": question,
            "type": item.get("type"),
            "strategies": {},
        }

        for strategy in strategies:
            print(f"  Running {strategy}...", flush=True)

            try:
                result = ask(
                    question=question,
                    col=col,
                    embedder=embedder,
                    k=5,
                    strategy=strategy,
                    reranker=(
                        reranker
                        if strategy != "dense"
                        else None
                    ),
                    bm25=(
                        bm25
                        if strategy == "hybrid"
                        else None
                    ),
                )

                comparison["strategies"][strategy] = {
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "status": "success",
                }

            except Exception as error:
                comparison["strategies"][strategy] = {
                    "status": "error",
                    "error": str(error),
                }

        results.append(comparison)

    # Save results
    output_dir = ROOT / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        output_dir / f"comparison_{timestamp}.json"
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