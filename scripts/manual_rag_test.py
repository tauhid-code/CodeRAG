import json
from pathlib import Path

from app.pipeline.rag_pipeline import RAGPipeline


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPO_PATH = (
    PROJECT_ROOT
    / "data"
    / "repositories"
    / "ml_end_to_end_test"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "ml_end_to_end_test"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "manual_rag_test_results.json"
)


# Questions for manual testing
TEST_QUESTIONS = [
    {
        "id": "Q001",
        "question": (
            "What does the ModelTrainer class do, "
            "and which file implements it?"
        ),
        "expected_file": (
            "src/components/model_training.py"
        ),
        "answerable": True,
    },
    {
        "id": "Q002",
        "question": (
            "How does the data transformation component work, "
            "and which file implements it?"
        ),
        "expected_file": (
            "src/components/transformation.py"
        ),
        "answerable": True,
    },
    {
        "id": "Q003",
        "question": (
            "How does the prediction pipeline work, "
            "and which file implements it?"
        ),
        "expected_file": (
            "src/pipeline/prediction_pipeline.py"
        ),
        "answerable": True,
    },
    {
        "id": "Q004",
        "question": (
            "Which custom exception is defined in the project, "
            "and where is it implemented?"
        ),
        "expected_file": "src/exception.py",
        "answerable": True,
    },
    {
        "id": "Q005",
        "question": (
            "What is the production API endpoint URL "
            "deployed for this project?"
        ),
        "expected_file": None,
        "answerable": False,
    },
]


def main():
    if not REPO_PATH.is_dir():
        raise FileNotFoundError(
            f"Repository directory not found: {REPO_PATH}"
        )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("CODERAG MANUAL TEST")
    print("=" * 65)

    # Initialize the existing pipeline.
    pipeline = RAGPipeline()

    # Index the repository and prepare retrieval.
    index_summary = pipeline.index_repository(
        repo_path=REPO_PATH,
        index_dir=INDEX_DIR,
    )

    print("\nINDEX SUMMARY")
    print(json.dumps(index_summary, indent=2))

    results = []

    # Ask every test question.
    for test_case in TEST_QUESTIONS:
        question_id = test_case["id"]
        question = test_case["question"]

        print("\n" + "=" * 65)
        print(f"{question_id}: {question}")
        print("=" * 65)

        try:
            response = pipeline.ask(question)

            sources = response.get("sources", [])

            source_results = []

            for source in sources:
                source_results.append(
                    {
                        "rank": source["rank"],
                        "file_path": source["file_path"],
                        "symbol_name": source["symbol_name"],
                        "symbol_type": source["symbol_type"],
                        "start_line": source["start_line"],
                        "end_line": source["end_line"],
                        "rerank_score": source["rerank_score"],
                        "content": source["content"],
                    }
                )

            expected_file = test_case["expected_file"]

            expected_file_retrieved = None

            if expected_file is not None:
                expected_file_retrieved = any(
                    expected_file.lower()
                    in source["file_path"].replace("\\", "/").lower()
                    for source in source_results
                )

            result = {
                "id": question_id,
                "question": question,
                "expected_file": expected_file,
                "answerable": test_case["answerable"],
                "answer": response.get("answer", ""),
                "sources": source_results,
                "expected_file_retrieved": expected_file_retrieved,
                "error": None,
            }

            print("\nANSWER:")
            print(result["answer"])

            print("\nRETRIEVED SOURCES:")

            if not source_results:
                print("No sources returned.")

            for source in source_results:
                print(
                    f"\nRank: {source['rank']}"
                    f"\nFile: {source['file_path']}"
                    f"\nSymbol: {source['symbol_name']}"
                    f"\nLines: {source['start_line']}-"
                    f"{source['end_line']}"
                    f"\nRerank score: {source['rerank_score']}"
                )

            print(
                "\nExpected file retrieved:",
                expected_file_retrieved,
            )

        except Exception as exc:
            result = {
                "id": question_id,
                "question": question,
                "expected_file": test_case["expected_file"],
                "answerable": test_case["answerable"],
                "answer": None,
                "sources": [],
                "expected_file_retrieved": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

            print("\nERROR:")
            print(result["error"])

        results.append(result)

        # Save after each question so completed results are preserved.
        with open(
            RESULTS_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "repository": str(REPO_PATH),
                    "index_summary": index_summary,
                    "results": results,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

    print("\n" + "=" * 65)
    print("MANUAL TEST COMPLETED")
    print("=" * 65)
    print(f"Questions attempted: {len(results)}")
    print(
        "Questions with errors:",
        sum(1 for result in results if result["error"] is not None),
    )
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
