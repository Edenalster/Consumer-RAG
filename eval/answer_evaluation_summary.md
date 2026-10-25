# Consumer-RAG — Answer Evaluation Summary

## Evaluation Setup

- Questions: 18
- Retrieval strategy: `fusion_llm_boosted`
- Generator: Gemini
- Evaluator: `gemini-3.5-flash-lite`
- Evaluation type: Automated LLM-as-a-Judge

## Results

| Metric | PASS | PARTIAL | FAIL | PASS Rate |
|---|---:|---:|---:|---:|
| faithfulness | 18 | 0 | 0 | 100.0% |
| citation_accuracy | 18 | 0 | 0 | 100.0% |
| correctness | 18 | 0 | 0 | 100.0% |
| completeness | 18 | 0 | 0 | 100.0% |
| abstention | 18 | 0 | 0 | 100.0% |

## Per-question Results

| ID | Faithfulness | Citations | Correctness | Completeness | Abstention |
|---|---|---|---|---|---|
| H01 | PASS | PASS | PASS | PASS | PASS |
| H02 | PASS | PASS | PASS | PASS | PASS |
| H03 | PASS | PASS | PASS | PASS | PASS |
| H04 | PASS | PASS | PASS | PASS | PASS |
| H05 | PASS | PASS | PASS | PASS | PASS |
| H06 | PASS | PASS | PASS | PASS | PASS |
| H07 | PASS | PASS | PASS | PASS | PASS |
| H08 | PASS | PASS | PASS | PASS | PASS |
| H09 | PASS | PASS | PASS | PASS | PASS |
| H10 | PASS | PASS | PASS | PASS | PASS |
| H11 | PASS | PASS | PASS | PASS | PASS |
| H12 | PASS | PASS | PASS | PASS | PASS |
| H13 | PASS | PASS | PASS | PASS | PASS |
| H14 | PASS | PASS | PASS | PASS | PASS |
| H15 | PASS | PASS | PASS | PASS | PASS |
| H16 | PASS | PASS | PASS | PASS | PASS |
| H17 | PASS | PASS | PASS | PASS | PASS |
| H18 | PASS | PASS | PASS | PASS | PASS |

## Limitations

- Scores are produced by an LLM Judge, not independent legal experts.
- The generator and judge use the same model family, which may introduce correlated errors.
- Faithfulness and citation evaluation use cited sources, not the full retrieved context.
- Correctness and completeness are evaluated against manually labeled ground-truth chunks.
- The evaluation set is small (18 questions) and was used during development.
- Ground-truth relevance labels and answerability classifications may be incomplete.
- Automated PASS does not establish legal accuracy or real-world reliability.

## Manual Review Findings

- H02: Correctly distinguishes optional registered mail from a legal requirement.
- H14: Appropriate abstention, but includes potentially distracting litigation advice.
- H18: Explains the general remote-sale definition while avoiding an unsupported conclusion about store pickup.

## Deterministic Citation Reference Validation

This check verifies that each answer contains at least one recognized citation and that all recognized citation numbers exist in its supplied source list.

- Valid answers: **18/18**
- Invalid answers: **0/18**
- Unit tests: **4 passed**

| Question | Reference validity |
|---|---|
| H01 | PASS |
| H02 | PASS |
| H03 | PASS |
| H04 | PASS |
| H05 | PASS |
| H06 | PASS |
| H07 | PASS |
| H08 | PASS |
| H09 | PASS |
| H10 | PASS |
| H11 | PASS |
| H12 | PASS |
| H13 | PASS |
| H14 | PASS |
| H15 | PASS |
| H16 | PASS |
| H17 | PASS |
| H18 | PASS |

**Scope limitation:** This validator checks recognized citation references only. It does not verify whether a cited source supports a claim, whether every claim has a citation, or whether citation syntax is malformed.

A negative control using a nonexistent `[מקור 99]` reference was correctly rejected by the deterministic validator, although the LLM citation judge accepted it.
