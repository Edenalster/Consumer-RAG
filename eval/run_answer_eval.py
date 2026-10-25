import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.consumer_rag.retrieval import build_index
from src.consumer_rag.reranking import Reranker
from src.consumer_rag.rag import ask


QUESTIONS_PATH = Path("eval/holdout_questions.jsonl")
OUTPUT_PATH = Path("eval/holdout_answers.jsonl")
STRATEGY = "fusion_llm_boosted"


def read_jsonl(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    load_dotenv(dotenv_path=".env")

    questions = read_jsonl(QUESTIONS_PATH)
    previous = read_jsonl(OUTPUT_PATH)

    completed = {
        record["id"]
        for record in previous
        if record.get("status") == "success"
        and record.get("strategy") == STRATEGY
    }

    pending = [
        question
        for question in questions
        if question["id"] not in completed
    ]

    if args.limit < 0:
        parser.error("--limit must be zero or greater")

    pending = pending[:args.limit]

    print("Total questions:", len(questions))
    print("Already completed:", len(completed))
    print("Running now:", len(pending))

    if not pending:
        print("Nothing to run.")
        return

    col, embedder, _ = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    reranker = Reranker()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for question in pending:
        qid = question["id"]
        print(f"\nRunning {qid}...")

        try:
            result = ask(
                question["q"],
                col,
                embedder,
                k=5,
                strategy=STRATEGY,
                reranker=reranker,
            )

            record = {
                "id": qid,
                "question": question["q"],
                "strategy": STRATEGY,
                "status": "success",
                "answer": result["answer"],
                "sources": result["sources"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            with OUTPUT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"Saved {qid} | "
                f"Cited sources: {len(result['sources'])}"
            )

        except Exception as error:
            print(f"FAILED {qid}: {type(error).__name__}: {error}")
            print("Stopping. Successful answers remain saved.")
            raise

    print("\nDONE")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
