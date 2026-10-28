"""Turn uploaded files into page-aware text chunks."""

from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class UnsupportedFile(ValueError):
    pass


def load_file(filename: str, data: bytes) -> list[Document]:
    """One Document per PDF page (one per text file), tagged with its source and page number."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFile(f"{filename}: only PDF, TXT and MD files are supported")

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        pages = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
    else:
        pages = [(1, data.decode("utf-8", errors="replace"))]

    return [
        Document(page_content=text, metadata={"source": filename, "page": number})
        for number, text in pages
        if text.strip()
    ]


def split(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)
