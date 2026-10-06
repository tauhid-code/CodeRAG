
from sentence_transformers import CrossEncoder

from app.retrieval.hybrid_retriever import HybridResult


class CodeReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device="cpu",
        )
        self.batch_size = batch_size

    @staticmethod
    def _format_chunk(chunk) -> str:
        """Prepare code and metadata for the reranker."""
        return (
            f"File: {chunk.file_path}\n"
            f"Symbol: {chunk.symbol_name}\n"
            f"Type: {chunk.symbol_type}\n"
            f"Code:\n{chunk.content}"
        )

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        top_k: int | None = None,
    ) -> list[HybridResult]:
        """Rerank hybrid retrieval results by cross-encoder relevance."""
        if not query.strip():
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not results:
            return []

        # Prepare query-document pairs.
        pairs = [
            (query, self._format_chunk(result.chunk))
            for result in results
        ]

        # Calculate cross-encoder relevance scores.
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Attach reranking scores while preserving RRF scores.
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        # Sort by cross-encoder relevance.
        reranked = sorted(
            results,
            key=lambda result: result.rerank_score,
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked