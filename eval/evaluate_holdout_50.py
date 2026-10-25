import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.consumer_rag.retrieval import build_index, search
from src.consumer_rag.reranking import Reranker
from src.consumer_rag.hybrid import reciprocal_rank_fusion
from src.consumer_rag.legal_classifier import classify_legal_query
from src.consumer_rag.category_boost import boost_by_category_linear


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def retrieve(question, col, embedder, reranker):
    category = classify_legal_query(question)

    if category == "לא ידוע" or category in question:
        expanded_question = question
    else:
        expanded_question = f"{question} {category}"

    dense_results = search(
        col, embedder, expanded_question, k=20
    )

    reranked_results = reranker.rerank(
        expanded_question, dense_results
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        reranked_results,
        dense_weight=0.4,
        sparse_weight=0.6,
    )

    boosted_results = boost_by_category_linear(
        fused_results,
        category,
        boost=10,
    )

    return boosted_results[:5], category


def main():
    load_dotenv(dotenv_path=".env")

    questions = (
        load_jsonl("eval/holdout_questions.jsonl")
        + load_jsonl("eval/holdout_questions_new.jsonl")
    )

    ground_truth = (
        load_jsonl("eval/holdout_ground_truth.jsonl")
        + load_jsonl("eval/holdout_ground_truth_new.jsonl")
    )

    gt_by_id = {item["id"]: item for item in ground_truth}

    assert len(questions) == 50
    assert len(gt_by_id) == 50
    assert {q["id"] for q in questions} == set(gt_by_id)

    print("Loading retrieval models...", flush=True)

    col, embedder, n_chunks = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    reranker = Reranker()
    records = []

    for index, item in enumerate(questions, start=1):
        qid = item["id"]
        truth = gt_by_id[qid]
        relevant = set(truth["relevant"])

        results, category = retrieve(
            item["q"], col, embedder, reranker
        )

        retrieved_ids = [chunk["id"] for chunk in results]

        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, 1)
                if chunk_id in relevant
            ),
            None,
        )

        hits = len(relevant.intersection(retrieved_ids))

        record = {
            "id": qid,
            "question": item["q"],
            "answerability": truth["answerability"],
            "category": category,
            "relevant": sorted(relevant),
            "retrieved": retrieved_ids,
            "first_relevant_rank": first_rank,
            "hit_rate_at_1": int(first_rank == 1),
            "hit_rate_at_3": int(
                first_rank is not None and first_rank <= 3
            ),
            "hit_rate_at_5": int(first_rank is not None),
            "recall_at_5": (
                hits / len(relevant) if relevant else None
            ),
            "reciprocal_rank": (
                1 / first_rank if first_rank else 0
            ),
        }

        records.append(record)

        print(
            f"[{index:02}/50] {qid} | "
            f"answerability={truth['answerability']} | "
            f"rank={first_rank}",
            flush=True,
        )

    evaluable = [r for r in records if r["relevant"]]

    def average(key):
        return sum(r[key] for r in evaluable) / len(evaluable)

    metrics = {
        "hit_rate_at_1": average("hit_rate_at_1"),
        "hit_rate_at_3": average("hit_rate_at_3"),
        "hit_rate_at_5": average("hit_rate_at_5"),
        "recall_at_5": average("recall_at_5"),
        "mrr_at_5": average("reciprocal_rank"),
    }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": "fusion_llm_boosted",
        "total_questions": len(questions),
        "indexed_chunks": n_chunks,
        "retrieval_evaluable": len(evaluable),
        "no_labeled_relevant_chunks": len(records) - len(evaluable),
        "answerability_distribution": dict(
            Counter(r["answerability"] for r in records)
        ),
        "metrics": metrics,
        "questions": records,
    }

    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_path = output_dir / f"holdout_50_{timestamp}.json"

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RETRIEVAL EVALUATION ===")
    print(f"Questions: {len(records)}")
    print(f"Retrieval-evaluable: {len(evaluable)}")
    print(f"No labeled relevant chunks: {len(records) - len(evaluable)}")

    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
