from app.pipeline.rag_pipeline import RAGPipeline
from app.ingestion.github_loader import GitHubRepositoryLoader


repo_url = "https://github.com/tauhid-code/AI-Powered-Intelligent-Resume-Analyzer"

pipeline = RAGPipeline()

loader = GitHubRepositoryLoader()

repo_path = loader.clone_repository(
    repo_url,
    "hirelytics_ask_test",
)

result = pipeline.index_repository(
    repo_path,
    "data/indexes/ask_test",
)

print("\n===== INDEXING RESULT =====")
print(result)

result = pipeline.ask(
    "What preprocessing steps are performed on the resume and job description text?"
)

print("\n===== ANSWER =====")
print(result["answer"])

print("\n===== SOURCES =====")

for source in result["sources"]:
    print(
        source["rank"],
        source["file_path"],
        source["symbol_name"],
        "rerank=",
        source["rerank_score"],
    )