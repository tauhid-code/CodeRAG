from pathlib import Path
from git import Repo
import shutil


class GitHubRepositoryLoader:

    def __init__(self, base_path="data/repositories"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def clone_repository(self, repo_url: str, repo_name: str):
        repo_path = self.base_path / repo_name

        # Remove old copy if it already exists
        if repo_path.exists():
            shutil.rmtree(repo_path)

        print(f"Cloning repository: {repo_url}")

        Repo.clone_from(
            repo_url,
            repo_path
        )

        print(f"Repository cloned to: {repo_path}")

        return repo_path

    def get_python_files(self, repo_path: Path):

        python_files = []

        ignored_directories = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".idea",
            ".vscode"
        }

        for file_path in repo_path.rglob("*.py"):

            # Check whether any parent directory is ignored
            if any(
                part in ignored_directories
                for part in file_path.parts
            ):
                continue

            python_files.append(file_path)

        return python_files