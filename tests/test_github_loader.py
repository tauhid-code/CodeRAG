from app.ingestion.github_loader import GitHubRepositoryLoader


loader = GitHubRepositoryLoader()

repo_url = "https://github.com/tauhid-code/AI-Powered-Intelligent-Resume-Analyzer"
repo_name = "AI-Powered-Intelligent-Resume-Analyzer"

repo_path = loader.clone_repository(
    repo_url,
    repo_name
)

python_files = loader.get_python_files(repo_path)

print(f"\nFound {len(python_files)} Python files:\n")

for file in python_files[:20]:
    print(file)