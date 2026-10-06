
from pathlib import Path

from app.ingestion.repository_indexer import RepositoryIndexer
from app.retrieval.faiss_store import FAISSVectorStore


def main():
    repo_path = Path("data/repositories/Hirelytics")
    index_dir = Path("data/indexes/Hirelytics")

    # Step 1: Build the repository index
    indexer = RepositoryIndexer()

    indexer.index_repository(
        repo_path=repo_path,
        index_dir=index_dir
    )

    # Step 2: Verify that the saved files exist
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "chunks.pkl"

    assert index_path.exists(), "FAISS index missing."
    assert metadata_path.exists(), "Chunk metadata missing."

    print("\nBoth index files exist.")

    # Step 3: Load the saved index
    loaded_store = FAISSVectorStore(dimension=384)

    loaded_store.load(
        str(index_path),
        str(metadata_path)
    )

    # Step 4: Test retrieval from the loaded index
    query = "How are HTTP requests handled?"

    query_embedding = indexer.embedding_model.embed_text(
        query
    )

    results = loaded_store.search(
        query_embedding,
        top_k=3
    )

    print(f"\nQuery: {query}")
    print("\nTop retrieved chunks:")

    for rank, (chunk, score) in enumerate(
        results, start=1
    ):
        print(f"\nRank: {rank}")
        print(f"Similarity: {score:.4f}")
        print(f"File: {chunk.file_path}")
        print(f"Symbol: {chunk.symbol_name}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Code:\n{chunk.content[:500]}")

    print("\nIndexing and reload test completed.")


if __name__ == "__main__":
    main()