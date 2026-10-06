
from pathlib import Path
from collections import defaultdict

from app.embeddings.embedding_model import CodeEmbeddingModel
from app.retrieval.faiss_store import FAISSVectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CodeReranker


TOP_K = 5
CANDIDATE_K = 20

# Initial manually labelled evaluation queries.
# Add more questions and verify their relevant symbols
# as you inspect your repository.
EVALUATION_DATA = [
    {
        "query": "How is a JWT access token created?",
        "relevant_symbols": {"create_access_token"},
    },
    {
        "query": "How does the API authenticate a user using a bearer token?",
        "relevant_symbols": {"get_current_user"},
    },
    {
        "query": "How does a user log in?",
        "relevant_symbols": {"login"},
    },
    {
        "query": "How does a new user register?",
        "relevant_symbols": {"signup"},
    },
]


def get_chunk(result):
    """Extract a CodeChunk from different retrieval result formats."""
    if hasattr(result, "chunk"):
        return result.chunk

    if isinstance(result, tuple):
        return result[0]

    if isinstance(result, dict):
        return result["chunk"]

    return result


def get_symbol(result):
    chunk = get_chunk(result)
    return chunk.symbol_name


def calculate_metrics(results, relevant_symbols):
    """Calculate Precision@K, Recall@K and MRR."""
    symbols = [get_symbol(result) for result in results]

    hits = sum(
        1 for symbol in symbols
        if symbol in relevant_symbols
    )

    precision = hits / TOP_K
    recall = hits / len(relevant_symbols)

    reciprocal_rank = 0.0

    for rank, symbol in enumerate(symbols, start=1):
        if symbol in relevant_symbols:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
    }


def main():
    # 1. Load the existing FAISS index.
    index_dir = Path("data/indexes/Hirelytics")
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "chunks.pkl"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "FAISS index or metadata not found. "
            "Run repository indexing first."
        )

    faiss_store = FAISSVectorStore(dimension=384)
    faiss_store.load(
        str(index_path),
        str(metadata_path),
    )

    # 2. Build BM25 using the same code chunks.
    bm25_store = BM25Store()
    bm25_store.add_chunks(faiss_store.chunks)

    # 3. Initialize shared models and retrievers.
    embedding_model = CodeEmbeddingModel()

    hybrid_retriever = HybridRetriever(
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        embedding_model=embedding_model,
    )

    reranker = CodeReranker()

    # 4. Store metrics for each retrieval approach.
    methods = [
        "FAISS",
        "BM25",
        "Hybrid",
        "Hybrid + Reranker",
    ]

    totals = {
        method: defaultdict(float)
        for method in methods
    }

    # 5. Evaluate every query using all four methods.
    for item in EVALUATION_DATA:
        query = item["query"]
        relevant = item["relevant_symbols"]

        print("\n" + "=" * 65)
        print(f"QUERY: {query}")
        print(f"Relevant symbols: {sorted(relevant)}")
        print("=" * 65)

        # FAISS
        query_embedding = embedding_model.embed_text(query)
        faiss_results = faiss_store.search(
            query_embedding,
            top_k=TOP_K,
        )

        # BM25
        bm25_results = bm25_store.search(
            query,
            top_k=TOP_K,
        )

        # Hybrid retrieval
        hybrid_results = hybrid_retriever.retrieve(
            query=query,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

        # Hybrid retrieval followed by reranking.
        # Use a separate candidate list so the original
        # hybrid results remain unchanged.
        rerank_candidates = hybrid_retriever.retrieve(
            query=query,
            top_k=CANDIDATE_K,
            candidate_k=CANDIDATE_K,
        )

        reranked_results = reranker.rerank(
            query=query,
            results=rerank_candidates,
            top_k=TOP_K,
        )

        all_results = {
            "FAISS": faiss_results,
            "BM25": bm25_results,
            "Hybrid": hybrid_results,
            "Hybrid + Reranker": reranked_results,
        }

        for method, results in all_results.items():
            metrics = calculate_metrics(results, relevant)

            for metric, value in metrics.items():
                totals[method][metric] += value

            print(f"\n{method}")
            print(
                f"  Precision@{TOP_K}: "
                f"{metrics['precision']:.3f}"
            )
            print(
                f"  Recall@{TOP_K}: "
                f"{metrics['recall']:.3f}"
            )
            print(f"  MRR: {metrics['mrr']:.3f}")

            print("  Retrieved symbols:")
            for rank, result in enumerate(results, start=1):
                print(f"    {rank}. {get_symbol(result)}")

    # 6. Calculate averages across all evaluation queries.
    num_queries = len(EVALUATION_DATA)

    print("\n" + "=" * 65)
    print("FINAL AVERAGE COMPARISON")
    print("=" * 65)

    print(
        f"{'Method':<24}"
        f"{'Precision@5':>14}"
        f"{'Recall@5':>12}"
        f"{'MRR':>9}"
    )
    print("-" * 65)

    for method in methods:
        avg_precision = totals[method]["precision"] / num_queries
        avg_recall = totals[method]["recall"] / num_queries
        avg_mrr = totals[method]["mrr"] / num_queries

        print(
            f"{method:<24}"
            f"{avg_precision:>14.3f}"
            f"{avg_recall:>12.3f}"
            f"{avg_mrr:>9.3f}"
        )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()