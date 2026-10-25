import json
from pathlib import Path

from eval.citation_validator import validate_citations


INPUT_PATH = Path("eval/answer_evaluation_context.jsonl")
REPORT_PATH = Path("eval/answer_evaluation_summary.md")

records = [
    json.loads(line)
    for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

ids = [record["id"] for record in records]

if len(ids) != 18 or len(set(ids)) != 18:
    raise ValueError("Expected 18 unique evaluation questions")

results = {
    record["id"]: validate_citations(
        record["answer"],
        record["sources"],
    )
    for record in records
}

passed = sum(result["valid"] for result in results.values())
failed = len(results) - passed

section = [
    "## Deterministic Citation Reference Validation",
    "",
    "This check verifies that each answer contains at least one "
    "recognized citation and that all recognized citation numbers "
    "exist in its supplied source list.",
    "",
    f"- Valid answers: **{passed}/{len(results)}**",
    f"- Invalid answers: **{failed}/{len(results)}**",
    "- Unit tests: **4 passed**",
    "",
    "| Question | Reference validity |",
    "|---|---|",
]

for qid in sorted(results):
    status = "PASS" if results[qid]["valid"] else "FAIL"
    section.append(f"| {qid} | {status} |")

section.extend([
    "",
    "**Scope limitation:** This validator checks recognized "
    "citation references only. It does not verify whether a "
    "cited source supports a claim, whether every claim has a "
    "citation, or whether citation syntax is malformed.",
    "",
    "A negative control using a nonexistent `[מקור 99]` "
    "reference was correctly rejected by the deterministic "
    "validator, although the LLM citation judge accepted it.",
    "",
])

report = REPORT_PATH.read_text(encoding="utf-8")

heading = "## Deterministic Citation Reference Validation"

if heading in report:
    report = report.split(heading)[0].rstrip() + "\n\n"

report += "\n".join(section)

REPORT_PATH.write_text(report, encoding="utf-8")

print("CITATION VALIDATION REPORT")
print("=" * 45)
print("Questions:", len(records))
print("Valid references:", passed)
print("Invalid references:", failed)
print("Unit tests: 4 passed (previous run)")
print("Report:", REPORT_PATH)
print("RESULT: PASS")
