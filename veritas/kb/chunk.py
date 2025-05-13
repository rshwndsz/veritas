from dataclasses import dataclass
from typing import List, Optional, Protocol
from uuid import uuid4


class Chunker(Protocol):
    def chunks(self, text: str) -> list[str]: ...


@dataclass
class Chunk:
    doc_id: str
    source_url: str
    chunk_index: int
    content: str
    embedding: Optional[list[float]] = None


@dataclass
class Document:
    claim_id: str
    type: str
    query: str
    url: str
    url2text: list[str]


def clean_document(text):
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "…": "...",
    }
    for not_plain, plain in replacements.items():
        text = text.replace(not_plain, plain)

    # Replace null-bytes with the unicode replacement character
    text = text.replace("\x00", "\ufffd")

    sentences = [s.strip() for s in text.split('",') if s.strip()]
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        normalized = sentence.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)

    cleaned_text = ". ".join(unique_sentences) + '",'
    return cleaned_text


def chunk_document(document: Document, text_splitter: Chunker) -> List[Chunk]:
    # Return an empty list for empty articles
    if len(document.url2text) == 0:
        return []

    # Create a new UUID for this document
    doc_id = str(uuid4())

    # Split the full document into chunks using a semantic text splitter
    text = " ".join(document.url2text)
    text = clean_document(text)
    text_chunks = text_splitter.chunks(text)

    # Build chunk objects
    chunks = [
        Chunk(
            doc_id=doc_id,
            source_url=document.url,
            chunk_index=i,
            content=text,
        )
        for i, text in enumerate(text_chunks)
    ]
    return chunks
