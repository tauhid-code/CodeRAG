from pathlib import Path

from app.ingestion.github_loader import GitHubRepositoryLoader
from app.ingestion.code_chunker import PythonCodeChunker
from app.embeddings.embedding_model import CodeEmbeddingModel


def main():

    # Step 1: Repository

    repo_path = Path(
        "data/repositories/requests"
    )

    if not repo_path.exists():

        print("Repository not found.")

        print(
            "Run test_github_loader.py first."
        )

        return
    
    # Step 2: Find Python files
  

    loader = GitHubRepositoryLoader()

    python_files = loader.get_python_files(
        repo_path
    )

    print(
        f"Found {len(python_files)} "
        f"Python files."
    )

  
    chunker = PythonCodeChunker()


    target_file = None

    for file in python_files:

        if file.name == "models.py":

            target_file = file
            break

    if target_file is None:

        print("models.py not found.")

        return

    print(
        f"\nProcessing: {target_file}"
    )

    chunks = chunker.chunk_file(
        target_file
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # Load embedding model


    embedding_model = CodeEmbeddingModel()

    # Generate embeddings


    embeddings = (
        embedding_model.embed_chunks(
            chunks
        )
    )


    print(
        f"\nGenerated embeddings: "
        f"{len(embeddings)}"
    )

    if embeddings:

        print(
            f"Embedding dimension: "
            f"{len(embeddings[0])}"
        )

        print(
            "\nFirst embedding "
            "(first 10 values):"
        )

        print(
            embeddings[0][:10]
        )


if __name__ == "__main__":
    main()