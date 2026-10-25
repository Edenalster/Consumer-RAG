from eval.citation_validator import validate_citations


SOURCES = [
    {"number": 1},
    {"number": 2},
    {"number": 3},
]


def test_valid_citation():
    result = validate_citations(
        "ניתן לבטל את העסקה [מקור 1].",
        SOURCES,
    )
    assert result["valid"] is True
    assert result["invalid_numbers"] == []


def test_nonexistent_citation():
    result = validate_citations(
        "ניתן לבטל את העסקה [מקור 99].",
        SOURCES,
    )
    assert result["valid"] is False
    assert result["invalid_numbers"] == [99]


def test_missing_citations():
    result = validate_citations(
        "ניתן לבטל את העסקה.",
        SOURCES,
    )
    assert result["valid"] is False
    assert result["has_citations"] is False


def test_multiple_citations():
    result = validate_citations(
        "ניתן לבטל את העסקה [מקור 1, מקור 2].",
        SOURCES,
    )
    assert result["valid"] is True
    assert result["cited_numbers"] == [1, 2]


def test_malformed_citation_must_fail():
    result = validate_citations(
        "ניתן לבטל את העסקה [מקור 1, מקור X].",
        SOURCES,
    )
    assert result["valid"] is False


def test_malformed_citation_without_number_must_fail():
    result = validate_citations(
        "ניתן לבטל את העסקה [מקור X].",
        SOURCES,
    )
    assert result["valid"] is False
