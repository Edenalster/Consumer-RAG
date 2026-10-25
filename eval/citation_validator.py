import json
import re
from pathlib import Path


def validate_citations(answer, sources):
    valid_numbers = {
        int(source["number"])
        for source in sources
    }

    citation_groups = re.findall(
        r"\[([^\[\]]*מקור[^\[\]]*)\]",
        answer,
    )

    cited_numbers = []
    malformed_groups = []

    citation_pattern = re.compile(
        r"מקור\s*(\d+)(?:\s*,\s*מקור\s*(\d+))*"
    )

    for group in citation_groups:
        # Each citation group must contain only valid
        # citation references separated by commas.
        full_pattern = r"מקור\s*\d+(?:\s*,\s*מקור\s*\d+)*"

        if not re.fullmatch(full_pattern, group.strip()):
            malformed_groups.append(group)
            continue

        cited_numbers.extend(
            int(n)
            for n in re.findall(r"מקור\s*(\d+)", group)
        )

    invalid_numbers = sorted(
        set(cited_numbers) - valid_numbers
    )

    return {
        "valid": (
            bool(cited_numbers)
            and not invalid_numbers
            and not malformed_groups
        ),
        "cited_numbers": cited_numbers,
        "invalid_numbers": invalid_numbers,
        "malformed_groups": malformed_groups,
        "has_citations": bool(citation_groups),
    }


def main():
    path = Path("eval/answer_evaluation_context.jsonl")

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    passed = 0

    print("CITATION VALIDATOR")
    print("=" * 45)

    for record in records:
        result = validate_citations(
            record["answer"],
            record["sources"],
        )

        status = "PASS" if result["valid"] else "FAIL"

        print(
            f"{record['id']}: {status} "
            f"| invalid={result['invalid_numbers']} "
            f"| malformed={result['malformed_groups']}"
        )

        passed += result["valid"]

    print("\nValid citations:", passed, "/", len(records))


if __name__ == "__main__":
    main()
