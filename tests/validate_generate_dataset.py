
import json
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("data/evaluation/generation_question.json")

REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "reference_answer",
    "source_files",
    "expected_points",
    "answerable",
}

def validate_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError("Dataset must be a JSON list.")

    errors = []
    ids = []

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            errors.append(f"Question {index}: not an object")
            continue

        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            errors.append(
                f"Question {index}: missing {sorted(missing)}"
            )
            continue

        ids.append(item["id"])

        if not isinstance(item["question"], str) or not item["question"].strip():
            errors.append(f"{item['id']}: empty question")

        if not isinstance(item["reference_answer"], str) or not item["reference_answer"].strip():
            errors.append(f"{item['id']}: empty reference answer")

        if not isinstance(item["source_files"], list):
            errors.append(f"{item['id']}: source_files must be a list")

        if not isinstance(item["expected_points"], list):
            errors.append(f"{item['id']}: expected_points must be a list")

        if not isinstance(item["answerable"], bool):
            errors.append(f"{item['id']}: answerable must be boolean")

        if item["answerable"] and not item["source_files"]:
            errors.append(
                f"{item['id']}: answerable question has no source files"
            )

    if len(ids) != len(set(ids)):
        errors.append("Duplicate question IDs found.")

    answerable = sum(
        1 for q in questions
        if isinstance(q, dict) and q.get("answerable") is True
    )
    unanswerable = sum(
        1 for q in questions
        if isinstance(q, dict) and q.get("answerable") is False
    )

    if len(questions) != 100:
        errors.append(f"Expected 100 questions, found {len(questions)}")

    if answerable != 70:
        errors.append(f"Expected 70 answerable, found {answerable}")

    if unanswerable != 30:
        errors.append(f"Expected 30 unanswerable, found {unanswerable}")

    print(f"Total questions: {len(questions)}")
    print(f"Answerable: {answerable}")
    print(f"Unanswerable: {unanswerable}")

    categories = Counter(
        q["category"] for q in questions
        if isinstance(q, dict) and isinstance(q.get("category"), str)
    )
    print("\nQuestions by category:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nDataset structure validation passed!")

if __name__ == "__main__":
    validate_dataset()
