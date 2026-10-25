import re

from .retrieval import search
from .generation import generate_answer
from .reranking import Reranker, fused_search
from .context_expansion import expand_context
from .hybrid_retrieval import hybrid_search
from .legal_classifier import expand_query_with_llm
from .hybrid import reciprocal_rank_fusion
from .legal_classifier import classify_legal_query
from .category_boost import boost_by_category_linear


def extract_citations(answer: str, num_chunks: int) -> set[int]:
    """Extract valid source numbers from an LLM-generated answer."""

    citation_groups = re.findall(
        r"\[([^\[\]]*מקור\s+\d+[^\[\]]*)\]",
        answer,
    )

    cited_numbers = set()

    for group in citation_groups:
        numbers = re.findall(r"מקור\s+(\d+)", group)
        cited_numbers.update(int(number) for number in numbers)

    return {
        number
        for number in cited_numbers
        if 1 <= number <= num_chunks
    }


def ask(
    question: str,
    col,
    embedder,
    k: int = 5,
    strategy: str = "dense",
    reranker: Reranker | None = None,
    bm25=None,
) -> dict:
    """
    Run the RAG pipeline using the selected retrieval strategy.

    Supported strategies:
        - dense
        - fusion
        - fusion_expanded
        - fusion_llm_expanded
        - hybrid
        - fusion_llm_boosted
    """

    # --------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # --------------------------------------------------

    if strategy == "dense":

        chunks = search(
            col,
            embedder,
            question,
            k=k,
        )

    elif strategy in ("fusion", "fusion_expanded"):

        if reranker is None:
            raise ValueError(
                f"{strategy} strategy requires a reranker"
            )

        chunks = fused_search(
            col,
            embedder,
            reranker,
            question,
            k=k,
            candidate_k=20,
        )

        # Expand context only for fusion_expanded
        if strategy == "fusion_expanded":

            chunks = expand_context(
                col=col,
                embedder=embedder,
                query=question,
                base_results=chunks,
                extra_k=5,
            )

    elif strategy == "fusion_llm_expanded":

        if reranker is None:
            raise ValueError(
                "fusion_llm_expanded strategy requires a reranker"
            )

        # Classify and expand the query using Gemini
        expanded_question = expand_query_with_llm(
            question
        )

        # Retrieve candidates using the expanded query
        dense_results = search(
            col,
            embedder,
            expanded_question,
            k=20,
        )

        # Rerank using the ORIGINAL question
        reranked_results = reranker.rerank(
            question,
            dense_results,
        )

        # Fuse Dense and CrossEncoder rankings
        fused_results = reciprocal_rank_fusion(
            dense_results,
            reranked_results,
            dense_weight=0.4,
            sparse_weight=0.6,
        )

        # Keep the top-k chunks
        chunks = fused_results[:k]
    elif strategy == "fusion_llm_boosted":

        if reranker is None:
            raise ValueError(
                "fusion_llm_boosted strategy requires a reranker"
            )

        # Step 1: Classify the legal category
        category = classify_legal_query(question)

        # Step 2: Expand the query
        if category == "לא ידוע" or category in question:
            expanded_question = question
        else:
            expanded_question = f"{question} {category}"

        # Step 3: Dense Retrieval
        dense_results = search(
            col,
            embedder,
            expanded_question,
            k=20,
        )

        # Step 4: Rerank using the EXPANDED question
        reranked_results = reranker.rerank(
            expanded_question,
            dense_results,
        )

        # Step 5: Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            dense_results,
            reranked_results,
            dense_weight=0.4,
            sparse_weight=0.6,
        )

        # Step 6: Category Boosting
        boosted_results = boost_by_category_linear(
            fused_results,
            category,
            boost=10,
        )

        # Step 7: Keep the top-k chunks
        chunks = boosted_results[:k]

    elif strategy == "hybrid":

        if reranker is None:
            raise ValueError(
                "hybrid strategy requires a reranker"
            )

        if bm25 is None:
            raise ValueError(
                "hybrid strategy requires a BM25 retriever"
            )

        chunks = hybrid_search(
            question=question,
            col=col,
            embedder=embedder,
            bm25=bm25,
            reranker=reranker,
            dense_k=5,
            bm25_k=5,
            final_k=k,
        )

    else:

        raise ValueError(
            f"Unknown retrieval strategy: {strategy}"
        )

    # --------------------------------------------------
    # Step 2: Generate answer
    # --------------------------------------------------

    answer = generate_answer(
        question=question,
        chunks=chunks,
    )

    # --------------------------------------------------
    # Step 3: Extract citations
    # --------------------------------------------------

    cited_numbers = extract_citations(
        answer,
        len(chunks),
    )

    # --------------------------------------------------
    # Step 4: Return only cited sources
    # --------------------------------------------------

    sources = [
        {
            "number": i,
            "title": chunk["meta"]["title"],
            "url": chunk["meta"]["url"],
            "chunk_id": chunk["id"],
            "distance": chunk.get("distance"),
        }
        for i, chunk in enumerate(chunks, start=1)
        if i in cited_numbers
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }