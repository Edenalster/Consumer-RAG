import pytest

from src.consumer_rag.category_boost import boost_by_category_linear

def make_chunk(chunk_id, doc_id):
    return {
        "id": chunk_id,
        "meta": {
            "doc_id": doc_id,
        },
    }


def test_boost_zero_preserves_order():
    chunks = [
        make_chunk("a", "cancel-in-store"),
        make_chunk("b", "cancel-doorstep"),
    ]

    result = boost_by_category_linear(
        chunks,
        category="עסקה ברוכלות",
        boost=0,
    )

    assert [chunk["id"] for chunk in result] == ["a", "b"]


def test_unknown_category_preserves_order():
    chunks = [
        make_chunk("a", "cancel-in-store"),
        make_chunk("b", "cancel-doorstep"),
    ]

    result = boost_by_category_linear(
        chunks,
        category="לא ידוע",
        boost=10,
    )

    assert [chunk["id"] for chunk in result] == ["a", "b"]


def test_matching_category_gets_priority():
    chunks = [
        make_chunk("a", "cancel-in-store"),
        make_chunk("b", "cancel-doorstep"),
        make_chunk("c", "cancel-online"),
    ]

    result = boost_by_category_linear(
        chunks,
        category="עסקה ברוכלות",
        boost=3,
    )

    assert result[0]["id"] == "b"


def test_negative_boost_raises_error():
    with pytest.raises(ValueError):
        boost_by_category_linear(
            [],
            category="עסקה ברוכלות",
            boost=-1,
        )


def test_original_list_is_not_modified():
    chunks = [
        make_chunk("a", "cancel-in-store"),
        make_chunk("b", "cancel-doorstep"),
    ]

    original_order = [chunk["id"] for chunk in chunks]

    boost_by_category_linear(
        chunks,
        category="עסקה ברוכלות",
        boost=5,
    )

    assert [chunk["id"] for chunk in chunks] == original_order


def test_all_chunks_are_preserved():
    chunks = [
        make_chunk("a", "cancel-in-store"),
        make_chunk("b", "cancel-doorstep"),
        make_chunk("c", "cancel-online"),
    ]

    result = boost_by_category_linear(
        chunks,
        category="עסקה ברוכלות",
        boost=10,
    )

    assert len(result) == len(chunks)

    assert {chunk["id"] for chunk in result} == {
        chunk["id"] for chunk in chunks
    }