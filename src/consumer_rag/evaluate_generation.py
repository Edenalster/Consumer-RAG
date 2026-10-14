import json
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import load_questions
from .rag import ask
from .retrieval import build_index
from .fetch import ROOT


def main():
    questions = load_questions(str(ROOT / "eval" / "questions.jsonl"))

    # Start with only 3 questions to limit API usage
    questions = questions[:3]

    print("Building retrieval index...")

    col, embedder, count = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    print(f"Indexed {count} chunks")

    results = []

    for i, item in enumerate(questions, start=1):
        question = item["q"]

        print(f"\n[{i}/{len(questions)}] {question}")

        try:
            result = ask(question, col, embedder, k=5)

            results.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "status": "success",
            })

            print("Answer generated successfully.")

        except Exception as e:
            results.append({
                "question": question,
                "answer": None,
                "sources": [],
                "status": "error",
                "error": str(e),
            })

            print(f"Error: {e}")

    output_dir = ROOT / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"generation_{timestamp}.json"

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()