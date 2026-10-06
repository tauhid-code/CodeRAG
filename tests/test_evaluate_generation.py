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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_question.json"

)

INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "Hirelytics"

INDEX_PATH = INDEX_DIR / "index.faiss"

METADATA_PATH = INDEX_DIR / "chunks.pkl"

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"

RESULTS_PATH = RESULTS_DIR / "groq_results.jsonl"

TOP_K = 5

CANDIDATE_K = 20

EMBEDDING_DIMENSION = 384

# --------------------------------------------------

# Dataset

# --------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    """Load and perform basic validation of the dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON list.")
    required_fields = {
        "id",
        "question",
        "answerable",
        "reference_answer",
        "expected_points",
        "source_files",
    }
    seen_ids = set()
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(
                f"Record {index} must be a JSON object."
            )
        missing = required_fields - item.keys()
        if missing:
            raise ValueError(
                f"Record {index} is missing fields: "
                f"{sorted(missing)}"
            )
        if not isinstance(item["id"], str) or not item["id"].strip():
            raise ValueError(f"Invalid ID at record {index}.")
        if item["id"] in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {item['id']}"
            )
        seen_ids.add(item["id"])
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(
                f"Empty question for {item['id']}."
            )
        if not isinstance(item["answerable"], bool):
            raise ValueError(
                f"Invalid answerable value for {item['id']}."
            )
        if not isinstance(item["expected_points"], list):
            raise ValueError(
                f"expected_points must be a list for {item['id']}."
            )
        if not isinstance(item["source_files"], list):
            raise ValueError(
                f"source_files must be a list for {item['id']}."
            )
    print(f"Loaded {len(dataset)} evaluation questions.")
    return dataset

# --------------------------------------------------

# Initialize retrieval and generation components

# --------------------------------------------------

def initialize_pipeline():
    """Initialize the existing CodeRAG components."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "FAISS index or metadata not found.\n"
            f"Expected index: {INDEX_PATH}\n"
            f"Expected metadata: {METADATA_PATH}\n"
            "Run repository indexing first."
        )
    print("\nLoading FAISS index...")
    faiss_store = FAISSVectorStore(
        dimension=EMBEDDING_DIMENSION
    )
    faiss_store.load(
        str(INDEX_PATH),
        str(METADATA_PATH),
    )
    if not faiss_store.chunks:
        raise RuntimeError("FAISS index contains no chunks.")
    print("\nBuilding BM25 index...")
    bm25_store = BM25Store()
    bm25_store.add_chunks(faiss_store.chunks)
    print("\nLoading embedding model...")
    embedding_model = CodeEmbeddingModel()
    print("\nInitializing hybrid retriever...")
    retriever = HybridRetriever(
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        embedding_model=embedding_model,
    )
    print("\nLoading reranker...")
    reranker = CodeReranker()
    print("\nInitializing Groq...")
    llm = GroqLLMClient()
    print(f"Groq model: {llm.model}")
    return retriever, reranker, llm

# --------------------------------------------------

# Context construction

# --------------------------------------------------

def build_context(results) -> str:
    """Format retrieved chunks for the generation model."""
    context_parts = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        part = (
            f"--- Retrieved chunk {rank} ---\n"
            f"File: {chunk.file_path}\n"
            f"Symbol: {chunk.symbol_name}\n"
            f"Type: {chunk.symbol_type}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"RRF score: {result.score:.6f}\n"
            f"Reranker score: {result.rerank_score}\n"
            f"Code:\n{chunk.content}\n"
        )
        context_parts.append(part)
    return "\n".join(context_parts)

def serialize_chunks(results) -> list[dict]:
    """Convert retrieved chunks into JSON-serializable records."""
    chunks = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        chunks.append({
            "rank": rank,
            "file_path": chunk.file_path,
            "symbol_name": chunk.symbol_name,
            "symbol_type": chunk.symbol_type,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "rrf_score": result.score,
            "rerank_score": result.rerank_score,
            "content": chunk.content,
        })
    return chunks

# --------------------------------------------------

# Resume support

# --------------------------------------------------

def load_completed_ids(path: Path) -> set[str]:
    """Find questions with successfully saved Groq answers."""
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number} "
                    f"in {path}"
                ) from exc
            if record.get("status") == "success":
                completed.add(record["id"])
    return completed

# --------------------------------------------------

# Evaluate one question

# --------------------------------------------------

def evaluate_question(
    item: dict,
    retriever: HybridRetriever,
    reranker: CodeReranker,
    llm: GroqLLMClient,

) -> dict:
    """Retrieve context and generate one Groq answer."""
    question_id = item["id"]
    question = item["question"]
    print(f"\n{'=' * 65}")
    print(f"Question: {question_id}")
    print(question)
    print(f"Answerable: {item['answerable']}")
    print("=" * 65)
    record = {
        "id": question_id,
        "category": item.get("category"),
        "question": question,
        "answerable": item["answerable"],
        "reference_answer": item["reference_answer"],
        "expected_points": item["expected_points"],
        "source_files": item["source_files"],
        "retrieval": {
            "method": "hybrid_reranker",
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "chunks": [],
            "context": "",
        },
        "groq": {
            "model": llm.model,
            "answer": None,
            "latency_seconds": None,
        },
        "status": "failed",
        "error": None,
    }
    # Retrieve and rerank.
    results = retriever.retrieve(
        query=question,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K,
    )
    reranked_results = reranker.rerank(
        query=question,
        results=results,
        top_k=TOP_K,
    )
    context = build_context(reranked_results)
    record["retrieval"]["chunks"] = serialize_chunks(
        reranked_results
    )
    record["retrieval"]["context"] = context
    print(f"Retrieved chunks: {len(reranked_results)}")
    # Generate the answer.
    start_time = time.perf_counter()
    answer = llm.generate_answer(
        query=question,
        context=context,
    )
    latency = time.perf_counter() - start_time
    record["groq"]["answer"] = answer
    record["groq"]["latency_seconds"] = round(
        latency, 4
    )
    record["status"] = "success"
    print(f"\nGroq answer:\n{answer}")
    print(f"\nLatency: {latency:.2f} seconds")
    return record

# --------------------------------------------------

# Main evaluation

# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Groq generation evaluation for CodeRAG."
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First question number to process (inclusive).",
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last question number to process (inclusive).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of selected questions to process.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Process selected questions again, including those with "
             "previously successful results.",
    )

    args = parser.parse_args()

    # Keep the default run small to avoid unintentionally consuming API quota.
    if args.start is None and args.end is None and args.limit is None:
        start_id, end_id = 1, 5
    else:
        start_id = args.start if args.start is not None else 1
        end_id = args.end if args.end is not None else 100

    if start_id <= 0:
        parser.error("--start must be greater than zero.")
    if end_id <= 0:
        parser.error("--end must be greater than zero.")
    if end_id < start_id:
        parser.error("--end must be greater than or equal to --start.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")

    load_dotenv(PROJECT_ROOT / ".env")

    dataset = load_dataset(DATASET_PATH)

    selected = []
    for item in dataset:
        match = re.fullmatch(r"Q(\d+)", item["id"])
        if not match:
            raise ValueError(
                f"Question ID must use the format Q followed by digits: {item['id']}"
            )
        question_number = int(match.group(1))
        if start_id <= question_number <= end_id:
            selected.append(item)

    if args.limit is not None:
        selected = selected[:args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    completed = set()
    if not args.no_resume:
        completed = load_completed_ids(RESULTS_PATH)

    pending = [
        item for item in selected
        if item["id"] not in completed
    ]

    print(f"\nTotal dataset questions: {len(dataset)}")
    print(f"Selected question range: Q{start_id:03d}-Q{end_id:03d}")
    print(f"Selected questions: {len(selected)}")
    print(f"Already completed: {len(selected) - len(pending)}")
    print(f"Remaining: {len(pending)}")
    print(f"Output: {RESULTS_PATH}")

    if not pending:
        print("\nAll selected questions are already completed.")
        return

    retriever, reranker, llm = initialize_pipeline()

    successful = 0
    failed = 0

    with RESULTS_PATH.open("a", encoding="utf-8") as output:
        for item in pending:
            try:
                record = evaluate_question(
                    item=item,
                    retriever=retriever,
                    reranker=reranker,
                    llm=llm,
                )
                successful += 1

            except Exception as exc:
                failed += 1
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

                print(
                    f"\nERROR on {item['id']}: "
                    f"{type(exc).__name__}: {exc}"
                )

            # Save each result immediately so earlier results survive failures.
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"Saved result for {item['id']}.")

    print("\n" + "=" * 65)
    print("GENERATION EVALUATION COMPLETED")
    print("=" * 65)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
