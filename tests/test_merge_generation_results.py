
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "evaluation" / "results"

GEMINI_FILE = RESULTS / "gemini_results.jsonl"
GROQ_FILE = RESULTS / "groq_results.jsonl"
BACKUP_FILE = RESULTS / "groq_results_before_merge.jsonl.bak"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    gemini = load_jsonl(GEMINI_FILE)
    groq = load_jsonl(GROQ_FILE)

    # Preserve the original Groq file.
    shutil.copy2(GROQ_FILE, BACKUP_FILE)

    # First 67 records from Gemini.
    gemini_part = gemini[:67]

    # 33 records from Groq (zero-based indices 19 to 51).
    groq_part = groq[19:53]

    if len(gemini_part) != 67:
        raise ValueError(
            f"Expected 67 Gemini records, found {len(gemini_part)}"
        )

    if len(groq_part) != 33:
        raise ValueError(
            f"Expected 33 Groq records, found {len(groq_part)}"
        )

    # Combine without filtering or removing any records.
    combined = gemini_part + groq_part

    if len(combined) != 100:
        raise ValueError(f"Expected 100 records, got {len(combined)}")

    temp_file = GROQ_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    temp_file.replace(GROQ_FILE)

    print("Merge completed successfully!")
    print(f"Gemini records: {len(gemini_part)}")
    print(f"Groq records:   {len(groq_part)}")
    print(f"Total records:  {len(combined)}")
    print(f"Final file: {GROQ_FILE}")
    print(f"Backup: {BACKUP_FILE}")


if __name__ == "__main__":
    main()
