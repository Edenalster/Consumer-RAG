import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .retrieval import ROOT

STRATEGIES = ["dense", "fusion", "fusion_expanded"]
METRICS = ["correctness", "completeness"]


def load_question_types(path: Path) -> dict[int, str]:
    questions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return {
        index: question["type"]
        for index, question in enumerate(questions, start=1)
    }


def analyze_by_type(scores: list[dict], question_types: dict[int, str]) -> dict:
    grouped = defaultdict(list)

    for item in scores:
        question_id = item["question_id"]
        question_type = question_types[question_id]
        grouped[question_type].append(item)

    results = {}

    for question_type, items in grouped.items():
        results[question_type] = {
            "count": len(items),
            "strategies": {},
        }

        for strategy in STRATEGIES:
            metrics = {}

            for metric in METRICS:
                values = [
                    item["scores"][strategy][metric]
                    for item in items
                ]

                metrics[metric] = mean(values)

            metrics["overall"] = mean(
                metrics[metric] for metric in METRICS
            )

            results[question_type]["strategies"][strategy] = metrics

    return results


def main():
    scores_path = ROOT / "eval" / "manual_scores.json"
    questions_path = ROOT / "eval" / "questions.jsonl"

    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    question_types = load_question_types(questions_path)

    results = analyze_by_type(scores, question_types)

    print("\nGENERATION EVALUATION BY QUESTION TYPE")
    print("=" * 70)

    for question_type, data in sorted(results.items()):
        print(f"\n{question_type.upper()} ({data['count']} questions)")
        print("-" * 70)

        print(
            f"{'Strategy':<20}"
            f"{'Correctness':<15}"
            f"{'Completeness':<15}"
            f"{'Overall':<10}"
        )

        for strategy, metrics in data["strategies"].items():
            print(
                f"{strategy:<20}"
                f"{metrics['correctness']:<15.2f}"
                f"{metrics['completeness']:<15.2f}"
                f"{metrics['overall']:<10.2f}"
            )


if __name__ == "__main__":
    main()