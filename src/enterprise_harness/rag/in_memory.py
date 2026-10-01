from __future__ import annotations

import re
from typing import Any

from .models import Chunk, Document, RetrievalResult
from .retriever import Retriever


class InMemoryRetriever(Retriever):
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._documents: dict[str, Document] = {}

    def add_document(self, doc: Document, chunk_size: int = 500) -> None:
        self._documents[doc.doc_id] = doc

        paragraphs = re.split(r"\n\s*\n", doc.content)
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) > chunk_size:
                for j in range(0, len(paragraph), chunk_size):
                    sub = paragraph[j : j + chunk_size].strip()
                    if sub:
                        chunk_id = f"{doc.doc_id}-chunk-{i}-{j // chunk_size}"
                        self._chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                doc_id=doc.doc_id,
                                content=sub,
                                index=len(self._chunks),
                                metadata=dict(doc.metadata),
                            )
                        )
            else:
                chunk_id = f"{doc.doc_id}-chunk-{i}"
                self._chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        content=paragraph,
                        index=len(self._chunks),
                        metadata=dict(doc.metadata),
                    )
                )

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored: list[tuple[Chunk, float]] = []
        for chunk in self._chunks:
            content_lower = chunk.content.lower()

            term_matches = sum(
                1 for term in query_terms if term in content_lower
            )
            exact_match = 2.0 if query_lower in content_lower else 0.0

            score = term_matches + exact_match

            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for chunk, score in scored[:top_k]:
            doc = self._documents.get(chunk.doc_id)
            results.append(
                RetrievalResult(
                    content=chunk.content,
                    score=score,
                    source=doc.title if doc else chunk.doc_id,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "doc_title": doc.title if doc else "",
                        **chunk.metadata,
                    },
                )
            )

        return results

    def index_documents(self, docs: list[Document], chunk_size: int = 500) -> None:
        for doc in docs:
            self.add_document(doc, chunk_size=chunk_size)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)