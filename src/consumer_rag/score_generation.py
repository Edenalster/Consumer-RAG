import json
from pathlib import Path
from statistics import mean

from .retrieval import ROOT


STRATEGIES = ["dense", "fusion", "fusion_expanded"]
METRICS = ["correctness", "completeness"]


def load_scores(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def calculate_averages(data: list[dict]) -> dict:
    results = {}

    for strategy in STRATEGIES:
        results[strategy] = {}

        for metric in METRICS:
            values = [
                item["scores"][strategy][metric]
                for item in data
            ]

            results[strategy][metric] = mean(values)

        results[strategy]["overall"] = mean(
            results[strategy][metric]
            for metric in METRICS
        )

    return results


def main():
    scores_path = ROOT / "eval" / "manual_scores.json"
    data = load_scores(scores_path)

    averages = calculate_averages(data)

    print("\nGENERATION EVALUATION RESULTS")
    print("=" * 65)

    print(
        f"{'Strategy':<20}"
        f"{'Correctness':<15}"
        f"{'Completeness':<15}"
        f"{'Overall':<10}"
    )

    print("-" * 65)

    for strategy, metrics in averages.items():
        print(
            f"{strategy:<20}"
            f"{metrics['correctness']:<15.2f}"
            f"{metrics['completeness']:<15.2f}"
            f"{metrics['overall']:<10.2f}"
        )

    print("\nScores range from 0 to 2.")
    print(f"Questions evaluated: {len(data)}")


if __name__ == "__main__":
    main()