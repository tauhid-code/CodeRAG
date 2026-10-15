
import argparse
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from app.embeddings.embedding_model import CodeEmbeddingModel
from app.generation.groq_client import GroqLLMClient
from app.retrieval.bm25_store import BM25Store
from app.retrieval.faiss_store import FAISSVectorStore
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CodeReranker


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data" / "evaluation" / "generation_question.json"
INDEX_DIR = ROOT / "data" / "indexes" / "Hirelytics"
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "chunks.pkl"

RESULTS_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "results"
    / "groq_results_after_retrieval_fix.jsonl"
)

TOP_K = 5
CANDIDATE_K = 20


def expand_code_query(query: str) -> str:
    """Add code-oriented search terms based on question intent."""
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
        extra_terms.append("def function method implementation")

    if any(
        word in query_lower
        for word in ["class", "object", "defined"]
    ):
        extra_terms.append("class definition")

    if any(
        word in query_lower
        for word in ["file", "path", "located"]
    ):
        extra_terms.append("file path module")

    if extra_terms:
        return query + " " + " ".join(extra_terms)

    return query


def load_dataset() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("The question dataset must be a JSON list.")

    return dataset


def initialize_pipeline():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Index files not found in {INDEX_DIR}. "
            "Check your existing FAISS index location."
        )

    print("Loading FAISS index...")
    faiss_store = FAISSVectorStore(dimension=384)
    faiss_store.load(str(INDEX_PATH), str(METADATA_PATH))

    print("Building BM25...")
    bm25_store = BM25Store()
    bm25_store.add_chunks(faiss_store.chunks)

    print("Loading embedding model...")
    embedding_model = CodeEmbeddingModel()

    retriever = HybridRetriever(
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        embedding_model=embedding_model,
        rrf_k=60,
    )

    print("Loading reranker...")
    reranker = CodeReranker()

    print("Initializing Groq...")
    llm = GroqLLMClient()

    return retriever, reranker, llm


def serialize_chunks(results) -> list[dict]:
    chunks = []

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk

        chunks.append(
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

    return chunks


def build_context(results) -> str:
    context_parts = []

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk

        context_parts.append(
            f"--- Retrieved chunk {rank} ---\n"
            f"File: {chunk.file_path}\n"
            f"Symbol: {chunk.symbol_name}\n"
            f"Type: {chunk.symbol_type}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"RRF score: {result.score:.6f}\n"
            f"Reranker score: {result.rerank_score}\n"
            f"Code:\n{chunk.content}"
        )

    return "\n\n".join(context_parts)


def retrieve_and_rerank(question, retriever, reranker):
    expanded_query = expand_code_query(question)

    original_results = retriever.retrieve(
        query=question,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K,
    )

    expanded_results = retriever.retrieve(
        query=expanded_query,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K,
    )

    merged_results = {}

    for result in original_results + expanded_results:
        chunk = result.chunk
        key = (
            str(chunk.file_path),
            chunk.symbol_name,
            chunk.start_line,
        )

        previous = merged_results.get(key)

        if previous is None or result.score > previous.score:
            merged_results[key] = result

    candidates = sorted(
        merged_results.values(),
        key=lambda result: result.score,
        reverse=True,
    )[:CANDIDATE_K]

    return reranker.rerank(
        query=question,
        results=candidates,
        top_k=TOP_K,
    )


def evaluate_question(item, retriever, reranker, llm):
    question = item["question"]

    print(f"\nProcessing {item['id']}: {question}")

    results = retrieve_and_rerank(
        question=question,
        retriever=retriever,
        reranker=reranker,
    )

    context = build_context(results)

    record = {
        "id": item["id"],
        "category": item.get("category"),
        "question": question,
        "answerable": item["answerable"],
        "reference_answer": item["reference_answer"],
        "expected_points": item["expected_points"],
        "source_files": item["source_files"],
        "retrieval": {
            "method": "expanded_hybrid_reranker",
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "chunks": serialize_chunks(results),
            "context": context,
        },
        "groq": {
            "model": llm.model,
            "answer": None,
            "latency_seconds": None,
        },
        "status": "failed",
        "error": None,
    }

    start_time = time.perf_counter()

    answer = llm.generate_answer(
        query=question,
        context=context,
    )

    record["groq"]["answer"] = answer
    record["groq"]["latency_seconds"] = round(
        time.perf_counter() - start_time,
        4,
    )
    record["status"] = "success"

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate CodeRAG answers with updated retrieval."
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    if args.start < 1 or args.end < args.start:
        parser.error("Provide a valid inclusive question range.")

    load_dotenv(ROOT / ".env")

    dataset = load_dataset()
    selected = []

    for item in dataset:
        match = re.fullmatch(r"Q(\d+)", item["id"])

        if match is None:
            continue

        number = int(match.group(1))

        if args.start <= number <= args.end:
            selected.append(item)

    if not selected:
        raise ValueError("No questions found in the requested range.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume from the new file only; never read or modify the old baseline.
    completed = set()

    if RESULTS_PATH.exists():
        with RESULTS_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                saved = json.loads(line)

                if saved.get("status") == "success":
                    completed.add(saved["id"])

    pending = [
        item for item in selected
        if item["id"] not in completed
    ]

    print(f"Selected questions: {len(selected)}")
    print(f"Already completed in new file: {len(selected) - len(pending)}")
    print(f"To process: {len(pending)}")
    print(f"Output: {RESULTS_PATH}")

    if not pending:
        print("All selected questions already have successful results.")
        return

    retriever, reranker, llm = initialize_pipeline()

    with RESULTS_PATH.open("a", encoding="utf-8") as output:
        for item in pending:
            try:
                record = evaluate_question(
                    item,
                    retriever,
                    reranker,
                    llm,
                )
            except Exception as exc:
                record = {
                    "id": item["id"],
                    "category": item.get("category"),
                    "question": item["question"],
                    "answerable": item["answerable"],
                    "reference_answer": item["reference_answer"],
                    "expected_points": item["expected_points"],
                    "source_files": item["source_files"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }

                print(f"ERROR for {item['id']}: {record['error']}")

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output.flush()

            print(f"Saved {item['id']} ({record['status']}).")

    print("\nGeneration batch completed.")
    print(f"New results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
