import re

from .retrieval import search
from .generation import generate_answer

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


def ask(question: str, col, embedder, k: int = 5) -> dict:
    """
    Retrieve relevant chunks and generate a grounded answer.
    """

    # Step 1: Retrieve relevant chunks
    chunks = search(
        col=col,
        embedder=embedder,
        query=question,
        k=k,
    )

    # Step 2: Generate answer based on retrieved chunks
    answer = generate_answer(
        question=question,
        chunks=chunks,
    )

        # Step 3: Extract cited source numbers
    cited_numbers = extract_citations(answer, len(chunks))

    # Step 4: Return only sources cited by Gemini
    sources = [
        {
            "number": i,
            "title": chunk["meta"]["title"],
            "url": chunk["meta"]["url"],
            "chunk_id": chunk["id"],
            "distance": chunk["distance"],
        }
        for i, chunk in enumerate(chunks, start=1)
        if i in cited_numbers
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }