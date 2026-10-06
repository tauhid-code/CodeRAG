from pathlib import Path

from app.ingestion.github_loader import GitHubRepositoryLoader
from app.ingestion.code_chunker import PythonCodeChunker


def main():

    # --------------------------------------------------
    # Step 1: Load repository
    # --------------------------------------------------

    loader = GitHubRepositoryLoader()

    repo_path = Path(
        "data/repositories/requests"
    )

    if not repo_path.exists():

        print("Repository not found.")

        print(
            "Run test_github_loader.py first."
        )

        return

    # --------------------------------------------------
    # Step 2: Find ALL Python files
    # --------------------------------------------------

    python_files = loader.get_python_files(
        repo_path
    )

    print(
        f"Found {len(python_files)} "
        f"Python files."
    )

    # --------------------------------------------------
    # Step 3: Create chunker
    # --------------------------------------------------

    chunker = PythonCodeChunker()

    # --------------------------------------------------
    # Step 4: Chunk entire repository
    # --------------------------------------------------

    all_chunks = chunker.chunk_repository(
        repo_path,
        python_files
    )

    # --------------------------------------------------
    # Step 5: Display summary
    # --------------------------------------------------

    print("\n" + "=" * 70)

    print("CHUNKING SUMMARY")

    print("=" * 70)

    print(
        f"Python files: {len(python_files)}"
    )

    print(
        f"Total chunks: {len(all_chunks)}"
    )

    # --------------------------------------------------
    # Step 6: Display first 10 chunks
    # --------------------------------------------------

    print("\nFIRST 10 CHUNKS")

    print("=" * 70)

    for index, chunk in enumerate(
        all_chunks[:10],
        start=1
    ):

        print(
            f"\nChunk #{index}"
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