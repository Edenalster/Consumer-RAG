import json
from collections import Counter
from pathlib import Path


BASE_PATH = Path("eval/answer_judge_results.jsonl")
EXTENDED_PATH = Path("eval/extended_judge_results.jsonl")
OUTPUT_PATH = Path("eval/answer_evaluation_summary.md")

METRICS = [
    "faithfulness",
    "citation_accuracy",
    "correctness",
    "completeness",
    "abstention",
]


def load_results(path):
    records = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            qid = record["id"]

            if qid in records:
                raise ValueError(f"Duplicate ID in {path}: {qid}")

            records[qid] = record

    return records


base = load_results(BASE_PATH)
extended = load_results(EXTENDED_PATH)

expected = {f"H{i:02d}" for i in range(1, 19)}

if set(base) != expected:
    raise ValueError("Base Judge results are incomplete")

if set(extended) != expected:
    raise ValueError("Extended Judge results are incomplete")

merged = {}

for qid in sorted(expected):
    merged[qid] = {
        **base[qid],
        **extended[qid],
    }

lines = [
    "# Consumer-RAG — Answer Evaluation Summary",
    "",
    "## Evaluation Setup",
    "",
    "- Questions: 18",
    "- Retrieval strategy: `fusion_llm_boosted`",
    "- Generator: Gemini",
    "- Evaluator: `gemini-3.5-flash-lite`",
    "- Evaluation type: Automated LLM-as-a-Judge",
    "",
    "## Results",
    "",
    "| Metric | PASS | PARTIAL | FAIL | PASS Rate |",
    "|---|---:|---:|---:|---:|",
]

for metric in METRICS:
    scores = Counter(
        merged[qid][metric]["score"]
        for qid in sorted(expected)
    )

    passed = scores["PASS"]
    partial = scores["PARTIAL"]
    failed = scores["FAIL"]

    lines.append(
        f"| {metric} | {passed} | {partial} | "
        f"{failed} | {passed / len(expected):.1%} |"
    )

lines.extend([
    "",
    "## Per-question Results",
    "",
    "| ID | Faithfulness | Citations | Correctness | Completeness | Abstention |",
    "|---|---|---|---|---|---|",
])

for qid in sorted(expected):
    r = merged[qid]

    lines.append(
        f"| {qid} | "
        f"{r['faithfulness']['score']} | "
        f"{r['citation_accuracy']['score']} | "
        f"{r['correctness']['score']} | "
        f"{r['completeness']['score']} | "
        f"{r['abstention']['score']} |"
    )

lines.extend([
    "",
    "## Limitations",
    "",
    "- Scores are produced by an LLM Judge, not independent legal experts.",
    "- The generator and judge use the same model family, which may introduce correlated errors.",
    "- Faithfulness and citation evaluation use cited sources, not the full retrieved context.",
    "- Correctness and completeness are evaluated against manually labeled ground-truth chunks.",
    "- The evaluation set is small (18 questions) and was used during development.",
    "- Ground-truth relevance labels and answerability classifications may be incomplete.",
    "- Automated PASS does not establish legal accuracy or real-world reliability.",
    "",
    "## Manual Review Findings",
    "",
    "- H02: Correctly distinguishes optional registered mail from a legal requirement.",
    "- H14: Appropriate abstention, but includes potentially distracting litigation advice.",
    "- H18: Explains the general remote-sale definition while avoiding an unsupported conclusion about store pickup.",
    "",
])

OUTPUT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("ANSWER EVALUATION SUMMARY")
print("=" * 45)
print("Base results:", len(base))
print("Extended results:", len(extended))
print("Questions merged:", len(merged))
print("Metrics:", len(METRICS))
print("Output:", OUTPUT_PATH)
print("RESULT: PASS")
