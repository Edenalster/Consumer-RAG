import json
from pathlib import Path

from .retrieval import ROOT


STRATEGIES = ["dense", "fusion", "fusion_expanded"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    results_dir = ROOT / "eval" / "results"

    comparison_files = sorted(
        results_dir.glob("comparison_*.json")
    )

    if not comparison_files:
        raise FileNotFoundError(
            "No comparison results found."
        )

    # Use the latest comparison
    comparison_path = comparison_files[-1]

    results = load_json(comparison_path)

    rubric = load_json(
        ROOT / "eval" / "generation_rubric.json"
    )

    rubric_by_question = {
        item["question"]: item
        for item in rubric
    }

    print(f"\nEvaluation file: {comparison_path.name}")

    for index, result in enumerate(results, start=1):
        question = result["question"]

        print("\n" + "=" * 75)
        print(f"QUESTION {index}: {question}")
        print("=" * 75)

        criteria = rubric_by_question.get(question)

        if criteria is None:
            print("WARNING: No rubric found.")
            continue

        print("\nEXPECTED FACTS:")

        for fact_index, fact in enumerate(
            criteria["expected_facts"],
            start=1,
        ):
            print(f"  {fact_index}. {fact}")

        print("\nIMPORTANT DISTINCTIONS:")

        for distinction in criteria["important_distinctions"]:
            print(f"  - {distinction}")

        for strategy in STRATEGIES:
            data = result["strategies"].get(strategy)

            print(f"\n--- {strategy.upper()} ---")

            if data is None:
                print("No result available.")
                continue

            if data["status"] != "success":
                print("ERROR:", data.get("error"))
                continue

            print("\nANSWER:")
            print(data["answer"])

            print("\nCITED SOURCES:")

            for source in data["sources"]:
                print(
                    f"  [{source['number']}] "
                    f"{source['chunk_id']}"
                )


if __name__ == "__main__":
    main()