from pathlib import Path

from app.ingestion.github_loader import GitHubRepositoryLoader
from app.ingestion.code_chunker import PythonCodeChunker
from app.embeddings.embedding_model import CodeEmbeddingModel
from app.retrieval.faiss_store import FAISSVectorStore


def main():

    # --------------------------------------------------
    # Step 1: Repository
    # --------------------------------------------------

    repo_path = Path(
        "data/repositories/requests"
    )

    if not repo_path.exists():

        print("Repository not found.")

        return

    # --------------------------------------------------
    # Step 2: Find Python files
    # --------------------------------------------------

    loader = GitHubRepositoryLoader()

    python_files = loader.get_python_files(
        repo_path
    )

    # --------------------------------------------------
    # Step 3: Find models.py
    # --------------------------------------------------

    target_file = None

    for file in python_files:

        if file.name == "models.py":

            target_file = file
            break

    if target_file is None:

        print("models.py not found.")

        return

    # --------------------------------------------------
    # Step 4: Chunk the file
    # --------------------------------------------------

    chunker = PythonCodeChunker()

    chunks = chunker.chunk_file(
        target_file
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------------------
    # Step 5: Generate embeddings
    # --------------------------------------------------

    embedding_model = CodeEmbeddingModel()

    embeddings = (
        embedding_model.embed_chunks(
            chunks
        )
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    # --------------------------------------------------
    # Step 6: Create FAISS store
    # --------------------------------------------------

    vector_store = FAISSVectorStore(
        dimension=384
    )

    # --------------------------------------------------
    # Step 7: Add vectors
    # --------------------------------------------------

    vector_store.add_embeddings(
        embeddings,
        chunks
    )

    # --------------------------------------------------
    # Step 8: Create a test query
    # --------------------------------------------------

    query = (
        "How does the request object "
        "handle authentication?"
    )

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------
    # Step 9: Embed query
    # --------------------------------------------------

    query_embedding = (
        embedding_model.embed_text(
            query
        )
    )

    # --------------------------------------------------
    # Step 10: Search FAISS
    # --------------------------------------------------

    results = vector_store.search(
        query_embedding,
        top_k=5
    )

    # --------------------------------------------------
    # Step 11: Display results
    # --------------------------------------------------

    print(
        "\nTop retrieved chunks:"
    )

    print("=" * 70)

    for rank, (chunk, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank #{rank}"
        )

        print(
            f"Similarity: {score:.4f}"
        )

        print(
            f"File: {chunk.file_path}"
        )

        print(
            f"Symbol: {chunk.symbol_name}"
        )

        print(
            f"Type: {chunk.symbol_type}"
        )

        print(
            f"Lines: "
            f"{chunk.start_line}-"
            f"{chunk.end_line}"
        )

        print("\nCode:")

        print(chunk.content)

        print("-" * 70)


if __name__ == "__main__":
    main()