
from pathlib import Path

from app.embeddings.embedding_model import CodeEmbeddingModel
from app.retrieval.faiss_store import FAISSVectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CodeReranker


def main():
    # Step 1: Load the saved FAISS index.
    index_dir = Path("data/indexes/Hirelytics")
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "chunks.pkl"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "FAISS index or metadata not found. "
            "Run the repository indexing test first."
        )

    faiss_store = FAISSVectorStore(dimension=384)
    faiss_store.load(
        str(index_path),
        str(metadata_path),
    )

    # Step 2: Build the BM25 index.
    bm25_store = BM25Store()
    bm25_store.add_chunks(faiss_store.chunks)

    # Step 3: Initialize the embedding model.
    embedding_model = CodeEmbeddingModel()

    # Step 4: Initialize the hybrid retriever.
    retriever = HybridRetriever(
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        embedding_model=embedding_model,
    )

    # Step 5: Retrieve candidates using FAISS + BM25.
    query = "How does user authentication work?"

    hybrid_results = retriever.retrieve(
        query=query,
        top_k=10,
        candidate_k=20,
    )

    if not hybrid_results:
        raise RuntimeError("Hybrid retrieval returned no results.")

    print(f"\nQuery: {query}")
    print(f"Hybrid candidates: {len(hybrid_results)}")

    # Step 6: Initialize the cross-encoder reranker.
    reranker = CodeReranker()

    # Step 7: Rerank the retrieved candidates.
    results = reranker.rerank(
        query=query,
        results=hybrid_results,
        top_k=5,
    )

    # Step 8: Display the reranked results.
    print("\nReranked results:")

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk

        print(f"\nRank: {rank}")
        print(f"Reranker score: {result.rerank_score:.4f}")
        print(f"Original RRF score: {result.score:.6f}")
        print(f"FAISS rank: {result.faiss_rank}")
        print(f"BM25 rank: {result.bm25_rank}")
        print(f"File: {chunk.file_path}")
        print(f"Symbol: {chunk.symbol_name}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Code:\n{chunk.content[:400]}")

    # Step 9: Validate the reranking results.
    assert results, "Reranker returned no results"
    assert len(results) <= 5, "Returned more than top_k results"

    scores = [result.rerank_score for result in results]

    assert all(score is not None for score in scores), (
        "Some results have no reranker score"
    )

    assert scores == sorted(scores, reverse=True), (
        "Results are not sorted by reranker score"
    )

    assert all(result.score > 0 for result in results), (
        "Original RRF scores were not preserved"
    )

    assert all(result.chunk is not None for result in results)

    print("\nReranker test passed!")


if __name__ == "__main__":
    main()