# Holdout Retrieval Evaluation

## Setup

- Dataset: 9 consumer-rights documents
- Indexed chunks: 86
- Holdout questions: 18
- Evaluated questions: 17
- Unsupported question: H14
- Partially supported question: H18
- Dense model: multilingual-e5-small
- Reranker: BAAI/bge-reranker-v2-m3
- Fusion weights: 0.4 / 0.6
- Category boost: 10

## Initial Evaluation

| Metric | Fusion | LLM Boosted |
|---|---:|---:|
| Hit@1 | 0.706 | 0.706 |
| Hit@3 | 0.941 | 0.941 |
| Hit@5 | 0.941 | 0.941 |
| MRR@5 | 0.804 | 0.794 |

## Ground Truth Audit

H01 initially missed a relevant chunk:
cancel-doorstep::003

Manual inspection confirmed that this chunk directly
answers the question about cancellation eligibility.

The ground truth was updated after examining retrieval
results, so the revised metrics are diagnostic rather
than an independent holdout estimate.

## Revised Diagnostic Metrics

| Metric | Fusion | LLM Boosted |
|---|---:|---:|
| Hit@1 | 0.706 | 0.765 |
| Hit@3 | 0.941 | 1.000 |
| Hit@5 | 0.941 | 1.000 |
| MRR@5 | 0.804 | 0.853 |

## Error Analysis

- H01: Query expansion improved category retrieval.
  The initial ground truth omitted a relevant chunk.
- H17: Category boosting prioritized cancellation
  documents over the remote-sale definition.
- H18: Similar ranking issue. Available evidence
  provides only partial support for the pickup scenario.
- H14: Excluded from positive retrieval metrics
  because the corpus lacks explicit supporting evidence.

## Limitations

- Small evaluation set (17 answerable questions).
- Ground truth was partially revised after inspection.
- Hit@K measures whether at least one labeled relevant
  chunk appears in the top K results.
- Retrieval metrics do not measure generated-answer
  correctness or citation faithfulness.
- A fresh, untouched test set is required to validate
  future algorithm changes.
