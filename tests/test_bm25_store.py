
from app.ingestion.code_chunker import CodeChunk
from app.retrieval.bm25_store import BM25Store
def main():
    chunks = [
        CodeChunk(
            content=(
                "def create_access_token(data): "
                "return jwt.encode(data, SECRET_KEY)"
            ),
            file_path="backend/auth.py",
            symbol_name="create_access_token",
            symbol_type="function",
            start_line=10,
            end_line=11,
        ),
        CodeChunk(
            content=(
                "def login(payload): "
                "verify_password(payload.password)"
            ),
            file_path="backend/auth.py",
            symbol_name="login",
            symbol_type="function",
            start_line=20,
            end_line=21,
        ),
        CodeChunk(
            content=(
                "def parse_resume(file): "
                "extract text from PDF"
            ),
            file_path="backend/parser.py",
            symbol_name="parse_resume",
            symbol_type="function",
            start_line=5,
            end_line=6,
        ),
    ]

    store = BM25Store()
    store.add_chunks(chunks)

    query = "Where is the JWT access token created?"
    results = store.search(query, top_k=3)

    print(f"Query: {query}")
    print("\nBM25 results:")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"BM25 score: {score:.4f}")
        print(f"File: {chunk.file_path}")
        print(f"Symbol: {chunk.symbol_name}")
        print(f"Code: {chunk.content}")

    assert results, "BM25 returned no results"
    assert results[0][0].symbol_name == "create_access_token"

    print("\nBM25 test passed!")


if __name__ == "__main__":
    main()