from pathlib import Path

import pytest

from app.ingest import UnsupportedFile, load_file, split

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "northwind-employee-handbook.pdf"


def test_text_file_becomes_one_page():
    docs = load_file("notes.md", b"# Leave\nStaff get 24 days of paid leave.")
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "notes.md", "page": 1}


def test_pdf_pages_keep_their_numbers():
    docs = load_file(SAMPLE_PDF.name, SAMPLE_PDF.read_bytes())
    assert [d.metadata["page"] for d in docs] == list(range(1, len(docs) + 1))
    assert len(docs) > 1


def test_unsupported_file_is_rejected():
    with pytest.raises(UnsupportedFile):
        load_file("photo.png", b"not a document")


def test_long_text_is_split_and_keeps_metadata():
    chunks = split(load_file("long.txt", ("word " * 2000).encode()))
    assert len(chunks) > 1
    assert all(c.metadata == {"source": "long.txt", "page": 1} for c in chunks)
