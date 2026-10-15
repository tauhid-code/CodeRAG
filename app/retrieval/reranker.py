from sentence_transformers import CrossEncoder

from app.retrieval.hybrid_retriever import HybridResult


class CodeReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[HybridResult],
        top_k: int = 5,
    ) -> list[HybridResult]:
        if not query.strip() or not results or top_k <= 0:
            return []

        pairs = []

        for result in results:
            chunk = result.chunk

            document = (
                f"File: {chunk.file_path}\n"
                f"Symbol: {chunk.symbol_name}\n"
                f"Type: {chunk.symbol_type}\n"
                f"Code:\n{chunk.content}"
            )

            pairs.append((query, document))

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            result.rerank_score = float(score)
            reranked_results.append(result)

        reranked_results.sort(
            key=lambda result: result.rerank_score,
            reverse=True,
        )

        return reranked_results[:top_k]