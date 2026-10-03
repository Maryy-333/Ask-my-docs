"""Citation parser tests."""
from app.citations import extract_citations


def test_single_and_adjacent():
    assert extract_citations("Yes [1]. Also [2][3].", 4) == [1, 2, 3]


def test_comma_list():
    assert extract_citations("See [1, 3].", 4) == [1, 3]


def test_dedupes_and_sorts():
    assert extract_citations("[3] then [1] then [3]", 4) == [1, 3]


def test_out_of_range_ignored():
    assert extract_citations("[0] [5] [9]", 4) == []


def test_non_numeric_brackets_ignored():
    assert extract_citations("uses the [CLS] token", 4) == []


def test_no_citations():
    assert extract_citations("plain answer", 4) == []