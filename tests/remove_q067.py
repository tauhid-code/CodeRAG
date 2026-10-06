
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "data" / "evaluation" / "generation_question.json"

# Supports either extension.
RESULT_CANDIDATES = [
    ROOT / "data" / "evaluation" / "results" / "groq_results.json",
    ROOT / "data" / "evaluation" / "results" / "groq_results.jsonl",
]

QUESTION_ID = "Q067"


def read_records(path):
    """Read either a JSON array or JSONL file."""
    content = path.read_text(encoding="utf-8-sig").strip()

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"{path.name} must contain a JSON array or JSONL.")
        return data, "json"
    except json.JSONDecodeError:
        records = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]
        return records, "jsonl"


def write_records(path, records, fmt):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        if fmt == "json":
            json.dump(records, f, indent=2, ensure_ascii=False)
            f.write("\n")
        else:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def remove_question(path):
    records, fmt = read_records(path)
    original_count = len(records)

    filtered = [
        record for record in records
        if str(record.get("id", "")).strip().upper() != QUESTION_ID
    ]

    removed = original_count - len(filtered)

    if removed == 0:
        print(f"{path.name}: {QUESTION_ID} not found; unchanged.")
        return

    # Never overwrite an existing backup.
    backup = Path(str(path) + ".before_Q067_removal.bak")
    if backup.exists():
        raise FileExistsError(
            f"Backup already exists: {backup}\n"
            "Restore or rename it before running this script again."
        )

    shutil.copy2(path, backup)
    write_records(path, filtered, fmt)

    print(f"\nFile: {path.name}")
    print(f"Original records: {original_count}")
    print(f"Removed records: {removed}")
    print(f"Remaining records: {len(filtered)}")
    print(f"Backup: {backup}")


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    result_path = next(
        (p for p in RESULT_CANDIDATES if p.exists()),
        None,
    )

    if result_path is None:
        raise FileNotFoundError(
            "Could not find groq_results.json or groq_results.jsonl"
        )

    remove_question(DATASET)
    remove_question(result_path)

    print("\nQ067 removal completed.")


if __name__ == "__main__":
    main()
