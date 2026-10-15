import pytest

from src.consumer_rag.score_generation import calculate_averages


def test_calculate_averages():
    data = [
        {
            "question_id": 1,
            "scores": {
                "dense": {
                    "correctness": 2,
                    "completeness": 1,
                },
                "fusion": {
                    "correctness": 1,
                    "completeness": 1,
                },
                "fusion_expanded": {
                    "correctness": 2,
                    "completeness": 2,
                },
            },
        },
        {
            "question_id": 2,
            "scores": {
                "dense": {
                    "correctness": 1,
                    "completeness": 0,
                },
                "fusion": {
                    "correctness": 2,
                    "completeness": 1,
                },
                "fusion_expanded": {
                    "correctness": 1,
                    "completeness": 2,
                },
            },
        },
    ]

    results = calculate_averages(data)

    assert results["dense"]["correctness"] == pytest.approx(1.5)
    assert results["dense"]["completeness"] == pytest.approx(0.5)
    assert results["dense"]["overall"] == pytest.approx(1.0)

    assert results["fusion"]["correctness"] == pytest.approx(1.5)
    assert results["fusion"]["completeness"] == pytest.approx(1.0)
    assert results["fusion"]["overall"] == pytest.approx(1.25)

    assert results["fusion_expanded"]["correctness"] == pytest.approx(1.5)
    assert results["fusion_expanded"]["completeness"] == pytest.approx(2.0)
    assert results["fusion_expanded"]["overall"] == pytest.approx(1.75)