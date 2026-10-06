
from pathlib import Path

from app.ingestion.github_loader import GitHubRepositoryLoader
from app.ingestion.code_chunker import PythonCodeChunker
from app.embeddings.embedding_model import CodeEmbeddingModel
from app.retrieval.faiss_store import FAISSVectorStore


class RepositoryIndexer:
    def __init__(self):
        self.loader = GitHubRepositoryLoader()
        self.chunker = PythonCodeChunker()
        self.embedding_model = CodeEmbeddingModel()
        self.vector_store = None

    def index_repository(
        self,
        repo_path: str | Path,
        index_dir: str | Path,
    ):
        repo_path = Path(repo_path)
        index_dir = Path(index_dir)

        if not repo_path.is_dir():
            raise FileNotFoundError(
                f"Repository not found: {repo_path}"
            )

        # Step 1: Find Python files
        python_files = self.loader.get_python_files(
            repo_path
        )

        if not python_files:
            raise ValueError(
                "No eligible Python files found."
            )

        print(f"\nFound {len(python_files)} Python files.")

        # Step 2: Create a fresh FAISS index
        self.vector_store = FAISSVectorStore(
            dimension=384
        )

        total_chunks = 0
        failed_files = []

        # Step 3: Chunk and embed each file
        for number, file_path in enumerate(
            python_files, start=1
        ):
            print(
                f"\n[{number}/{len(python_files)}] "
                f"Processing: {file_path}"
            )

            try:
                chunks = self.chunker.chunk_file(
                    file_path
                )

                if not chunks:
                    print("No code chunks found; skipping.")
                    continue

                embeddings = self.embedding_model.embed_chunks(
                    chunks
                )

                self.vector_store.add_embeddings(
                    embeddings,
                    chunks
                )

                total_chunks += len(chunks)

                print(f"Created {len(chunks)} chunks.")

            except Exception as exc:
                failed_files.append(
                    (str(file_path), str(exc))
                )
                print(f"Failed to process: {exc}")

        # Step 4: Validate the index
        if self.vector_store.index.ntotal == 0:
            raise RuntimeError(
                "Indexing failed: no vectors were created."
            )

        # Step 5: Save the index and its metadata
        index_path = index_dir / "index.faiss"
        metadata_path = index_dir / "chunks.pkl"

        self.vector_store.save(
            str(index_path),
            str(metadata_path)
        )

        # Step 6: Print summary
        print("\n" + "=" * 60)
        print("REPOSITORY INDEXING COMPLETED")
        print("=" * 60)
        print(f"Python files found: {len(python_files)}")
        print(f"Chunks indexed: {total_chunks}")
        print(
            f"Vectors stored: "
            f"{self.vector_store.index.ntotal}"
        )
        print(f"Failed files: {len(failed_files)}")
        print(f"Index saved: {index_path}")
        print(f"Metadata saved: {metadata_path}")

        if failed_files:
            print("\nFiles that failed:")
            for path, error in failed_files:
                print(f"- {path}: {error}")

        return self.vector_store