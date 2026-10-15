
from pathlib import Path
from contextlib import redirect_stdout

from app.embeddings.embedding_model import CodeEmbeddingModel
from app.retrieval.faiss_store import FAISSVectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CodeReranker


INDEX_DIR = Path("data/indexes/ask_test")
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "chunks.pkl"

OUTPUT_PATH = Path("debug_ques23_output.txt")
QUERY = "Which model name is used as the default in the code?"
TARGET_FILE = "interview_coach.py"
TARGET_TEXT = "GROQ_MODEL"


def print_results(title, results):
    print(f"\n{'=' * 20} {title} {'=' * 20}\n")

    for rank, result in enumerate(results, start=1):
        if hasattr(result, "chunk"):
            chunk = result.chunk
            score = result.score
            rerank_score = result.rerank_score
            faiss_rank = result.faiss_rank
            bm25_rank = result.bm25_rank
        else:
            chunk, score = result
            rerank_score = None
            faiss_rank = None
            bm25_rank = None

        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"File: {chunk.file_path}")
        print(f"Symbol: {chunk.symbol_name}")
        print(f"Type: {chunk.symbol_type}")
        print(f"Lines: {chunk.start_line}-{chunk.end_line}")

        if faiss_rank is not None:
            print(f"FAISS rank: {faiss_rank}")

        if bm25_rank is not None:
            print(f"BM25 rank: {bm25_rank}")

        if rerank_score is not None:
            print(f"Reranker score: {rerank_score:.4f}")

        if TARGET_TEXT in chunk.content:
            print(">>> TARGET TEXT FOUND <<<")

        print("-" * 70)


def main():
    print("Q023 RETRIEVAL DIAGNOSTIC")
    print(f"Query: {QUERY}")

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        print("ERROR: Saved FAISS index or metadata not found.")
        print(f"Index: {INDEX_PATH}")
        print(f"Metadata: {METADATA_PATH}")
        return

    faiss_store = FAISSVectorStore()
    faiss_store.load(str(INDEX_PATH), str(METADATA_PATH))

    print(f"Loaded vectors: {faiss_store.index.ntotal}")
    print(f"Loaded chunks: {len(faiss_store.chunks)}")

    print("\nLoading embedding model...")
    embedding_model = CodeEmbeddingModel()

    query_embedding = embedding_model.embed_text(QUERY)
    faiss_results = faiss_store.search(query_embedding, top_k=10)
    print_results("FAISS TOP 10", faiss_results)

    bm25_store = BM25Store()
    bm25_store.add_chunks(faiss_store.chunks)
    bm25_results = bm25_store.search(QUERY, top_k=10)
    print_results("BM25 TOP 10", bm25_results)

    hybrid_retriever = HybridRetriever(
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        embedding_model=embedding_model,
        rrf_k=60,
    )

    hybrid_results = hybrid_retriever.retrieve(
        query=QUERY,
        top_k=10,
        candidate_k=20,
    )
    print_results("HYBRID RRF TOP 10", hybrid_results)

    print("\nLoading cross-encoder reranker...")
    reranker = CodeReranker()

    reranked_results = reranker.rerank(
        query=QUERY,
        results=hybrid_results,
        top_k=10,
    )
    print_results("HYBRID + RERANKER TOP 10", reranked_results)

    print("\nTARGET CHUNK CHECK")

    target_chunks = [
        chunk
        for chunk in faiss_store.chunks
        if TARGET_FILE in str(chunk.file_path)
        and TARGET_TEXT in chunk.content
    ]

    print(f"Chunks containing {TARGET_TEXT}: {len(target_chunks)}")

    for chunk in target_chunks:
        print(
            f"File={chunk.file_path} | "
            f"Symbol={chunk.symbol_name} | "
            f"Type={chunk.symbol_type} | "
            f"Lines={chunk.start_line}-{chunk.end_line}"
        )


if __name__ == "__main__":
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        errors="replace",
    ) as output_file:
        with redirect_stdout(output_file):
            main()

    print(f"Diagnostic saved to: {OUTPUT_PATH.resolve()}")
