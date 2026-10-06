
from dataclasses import dataclass
from typing import Any

from app.embeddings.embedding_model import CodeEmbeddingModel
from app.ingestion.code_chunker import CodeChunk
from app.retrieval.bm25_store import BM25Store
from app.retrieval.faiss_store import FAISSVectorStore


@dataclass
class HybridResult:
    chunk: CodeChunk
    score: float
    faiss_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None   


class HybridRetriever:
    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        bm25_store: BM25Store,
        embedding_model: CodeEmbeddingModel,
        rrf_k: int = 60,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.embedding_model = embedding_model
        self.rrf_k = rrf_k

    @staticmethod
    def _get_chunk(result: Any) -> CodeChunk:
        """Extract a CodeChunk from a retrieval result."""
        if isinstance(result, CodeChunk):
            return result

        if isinstance(result, tuple):
            return result[0]

        if isinstance(result, dict):
            return result["chunk"]

        if hasattr(result, "chunk"):
            return result.chunk

        raise TypeError(
            f"Unsupported retrieval result: {type(result)}"
        )

    @staticmethod
    def _chunk_key(chunk: CodeChunk) -> tuple:
        """Create a key to identify duplicate chunks."""
        return (
            chunk.file_path,
            chunk.symbol_name,
            chunk.symbol_type,
            chunk.start_line,
            chunk.end_line,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[HybridResult]:
        """Combine FAISS and BM25 rankings using RRF."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than zero"
            )

        if not query.strip():
            return []

        # 1. Semantic retrieval using FAISS.
        query_embedding = self.embedding_model.embed_text(query)
        faiss_results = self.faiss_store.search(
            query_embedding,
            top_k=candidate_k,
        )

        # 2. Keyword retrieval using BM25.
        bm25_results = self.bm25_store.search(
            query,
            top_k=candidate_k,
        )

        # 3. Combine rankings using Reciprocal Rank Fusion.
        combined = {}

        def add_to_ranking(results, source):
            for rank, result in enumerate(results, start=1):
                chunk = self._get_chunk(result)
                key = self._chunk_key(chunk)

                if key not in combined:
                    combined[key] = HybridResult(
                        chunk=chunk,
                        score=0.0,
                    )

                item = combined[key]
                item.score += 1.0 / (self.rrf_k + rank)

                if source == "faiss":
                    item.faiss_rank = rank
                else:
                    item.bm25_rank = rank

        add_to_ranking(faiss_results, "faiss")
        add_to_ranking(bm25_results, "bm25")

        # 4. Sort by the combined RRF score.
        ranked_results = sorted(
            combined.values(),
            key=lambda item: item.score,
            reverse=True,
        )

        # 5. Return the requested number of results.
        return ranked_results[:top_k]