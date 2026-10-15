import re
from typing import Iterable

from rank_bm25 import BM25Okapi

from app.ingestion.code_chunker import CodeChunk


class BM25Store:
    def __init__(self):
        self.chunks: list[CodeChunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25 = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize natural language and Python identifiers."""
        identifiers = re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*|\d+",
            text,
        )

        tokens = []

        for identifier in identifiers:
            normalized = identifier.lower()
            tokens.append(normalized)

            # Split camelCase and PascalCase identifiers.
            split_identifier = re.sub(
                r"([A-Z]+)([A-Z][a-z])",
                r"\1 \2",
                identifier,
            )
            split_identifier = re.sub(
                r"([a-z0-9])([A-Z])",
                r"\1 \2",
                split_identifier,
            )

            # Split snake_case and other separators.
            parts = re.split(
                r"[_\W]+",
                split_identifier.lower(),
            )

            for part in parts:
                if part and part != normalized:
                    tokens.append(part)

        return tokens

    def add_chunks(self, chunks: Iterable[CodeChunk]) -> None:
        """Build the BM25 index from the supplied chunks."""
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

        for chunk in chunks:
            # Include code and symbol metadata in lexical retrieval.
            searchable_text = (
                f"{chunk.file_path} "
                f"{chunk.symbol_name} "
                f"{chunk.symbol_type} "
                f"{chunk.content}"
            )

            tokens = self.tokenize(searchable_text)

            if not tokens:
                continue

            self.chunks.append(chunk)
            self.tokenized_corpus.append(tokens)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[CodeChunk, float]]:
        """Return the most relevant chunks and BM25 scores."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not query or not query.strip() or self.bm25 is None:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            results.append((self.chunks[index], score))

            if len(results) >= top_k:
                break

        return results
