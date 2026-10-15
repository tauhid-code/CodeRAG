
from pathlib import Path

from app.embeddings.embedding_model import CodeEmbeddingModel
from app.generation.groq_client import GroqLLMClient
from app.ingestion.repository_indexer import RepositoryIndexer
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CodeReranker


def expand_code_query(query: str) -> str:
    """Add code-oriented terms based on the question's intent."""
    query_lower = query.lower()
    extra_terms = []

    if any(
        word in query_lower
        for word in [
            "default",
            "configured",
            "configuration",
            "environment variable",
        ]
    ):
        extra_terms.append(
            "default os.getenv os.environ config settings"
        )

    if "model" in query_lower:
        extra_terms.append(
            "MODEL model= model_name GROQ_MODEL LLM_MODEL"
        )

    if any(
        word in query_lower
        for word in [
            "function",
            "method",
            "implemented",
            "implementation",
        ]
    ):
        extra_terms.append(
            "def function method implementation"
        )

    if any(
        word in query_lower
        for word in [
            "class",
            "object",
            "defined",
        ]
    ):
        extra_terms.append("class definition")

    if any(
        word in query_lower
        for word in [
            "file",
            "path",
            "located",
        ]
    ):
        extra_terms.append("file path module")

    if extra_terms:
        return query + " " + " ".join(extra_terms)

    return query


class RAGPipeline:
    def __init__(
        self,
        top_k: int = 5,
        candidate_k: int = 20,
    ):
        self.top_k = top_k
        self.candidate_k = candidate_k

        self.indexer = RepositoryIndexer()
        self.embedding_model = None
        self.faiss_store = None
        self.bm25_store = None
        self.hybrid_retriever = None
        self.reranker = None
        self.llm = None

        self.is_ready = False

    def index_repository(
        self,
        repo_path: str | Path,
        index_dir: str | Path,
    ):
        """Index a Python repository and prepare retrieval."""

        repo_path = Path(repo_path)
        index_dir = Path(index_dir)

        # 1. Build the FAISS index.
        self.faiss_store = self.indexer.index_repository(
            repo_path=repo_path,
            index_dir=index_dir,
        )

        # 2. Reuse the embedding model.
        self.embedding_model = self.indexer.embedding_model

        # 3. Build BM25 from the indexed chunks.
        self.bm25_store = BM25Store()
        self.bm25_store.add_chunks(
            self.faiss_store.chunks
        )

        # 4. Create the hybrid retriever.
        self.hybrid_retriever = HybridRetriever(
            faiss_store=self.faiss_store,
            bm25_store=self.bm25_store,
            embedding_model=self.embedding_model,
            rrf_k=60,
        )

        # 5. Create the reranker.
        self.reranker = CodeReranker()

        # 6. Create the Groq client.
        self.llm = GroqLLMClient()

        self.is_ready = True

        return {
            "python_files": len(
                self.indexer.loader.get_python_files(repo_path)
            ),
            "chunks": len(self.faiss_store.chunks),
            "vectors": self.faiss_store.index.ntotal,
        }

    def ask(self, query: str):
        """Retrieve, rerank and generate an answer."""

        if not self.is_ready:
            raise RuntimeError(
                "RAG pipeline is not ready. "
                "Index a repository first."
            )

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Expand the query for code-oriented retrieval.
        expanded_query = expand_code_query(query)

        # Retrieve candidates using the original question.
        original_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=self.candidate_k,
            candidate_k=self.candidate_k,
        )

        # Retrieve candidates using the expanded question.
        expanded_results = self.hybrid_retriever.retrieve(
            query=expanded_query,
            top_k=self.candidate_k,
            candidate_k=self.candidate_k,
        )

        # 2. Merge candidates and remove duplicate chunks.
        merged_results = {}

        for result in original_results + expanded_results:
            chunk = result.chunk

            chunk_key = (
                str(chunk.file_path),
                chunk.symbol_name,
                chunk.start_line,
            )

            existing_result = merged_results.get(chunk_key)

            if (
                existing_result is None
                or result.score > existing_result.score
            ):
                merged_results[chunk_key] = result

        hybrid_results = sorted(
            merged_results.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:self.candidate_k]

        if not hybrid_results:
            return {
                "answer": (
                    "I couldn't find relevant code in "
                    "the repository to answer this question."
                ),
                "sources": [],
            }

        # 3. Rerank the merged candidates using the original query.
        reranked_results = self.reranker.rerank(
            query=query,
            results=hybrid_results,
            top_k=self.top_k,
        )

        # 4. Build the context for the LLM.
        context_parts = []

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):
            chunk = result.chunk

            context_parts.append(
                f"Source {rank}:\n"
                f"File: {chunk.file_path}\n"
                f"Symbol: {chunk.symbol_name}\n"
                f"Type: {chunk.symbol_type}\n"
                f"Lines: {chunk.start_line}-{chunk.end_line}\n"
                f"Code:\n{chunk.content}"
            )

        context = "\n\n".join(context_parts)

        # 5. Generate the answer using the original question.
        answer = self.llm.generate_answer(
            query=query,
            context=context,
        )

        # 6. Prepare source information for the UI.
        sources = []

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):
            chunk = result.chunk

            sources.append(
                {
                    "rank": rank,
                    "file_path": str(chunk.file_path),
                    "symbol_name": chunk.symbol_name,
                    "symbol_type": chunk.symbol_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "rrf_score": result.score,
                    "rerank_score": result.rerank_score,
                    "content": chunk.content,
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }
