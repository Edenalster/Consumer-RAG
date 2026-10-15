from .retrieval import ROOT, build_index
from .reranking import Reranker, fused_search
from .evaluate import load_questions, first_hit_rank, summarize


CANDIDATE_VALUES = [5, 10, 20]


def main():
    questions = load_questions(
        str(ROOT / "eval" / "questions.jsonl")
    )

    print("Loading models and index...")

    col, embedder, _ = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    reranker = Reranker()

    print("\nCANDIDATE K RETRIEVAL EVALUATION")
    print("=" * 75)

    print(
        f"{'Candidate K':<15}"
        f"{'Hit@1':<12}"
        f"{'Hit@3':<12}"
        f"{'Hit@5':<12}"
        f"{'MRR':<12}"
    )

    print("-" * 75)

    for candidate_k in CANDIDATE_VALUES:
        ranks = []
        failed_questions = []

        for question in questions:
            results = fused_search(
                col,
                embedder,
                reranker,
                question["q"],
                k=5,
                candidate_k=candidate_k,
            )

            rank = first_hit_rank(
                results,
                doc=question.get("doc"),
                phrase=question.get("phrase"),
                relevant=question.get("relevant"),
            )

            ranks.append(rank)
            if rank is None or rank > 5:
                failed_questions.append(question["q"])

        metrics = summarize(ranks)
        

        print(
    f"{candidate_k:<15}"
    f"{metrics['recall@1']:<12.3f}"
    f"{metrics['recall@3']:<12.3f}"
    f"{metrics['recall@5']:<12.3f}"
    f"{metrics['mrr']:<12.3f}"
)
        if failed_questions:
            print(f"\nFailed questions for candidate_k={candidate_k}:")
        for question_text in failed_questions:
            print(f"  - {question_text}")
    else:
     print(f"\nNo failed questions for candidate_k={candidate_k}")


if __name__ == "__main__":
    main()