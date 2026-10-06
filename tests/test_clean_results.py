import json
import shutil
from pathlib import Path

path = Path("data/evaluation/results/gemini_results.jsonl")

# Create a backup first
backup = path.with_suffix(".jsonl.bak")
shutil.copy2(path, backup)

# Read and validate all records before changing the file
with path.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Keep Q001–Q069 and remove Q070–Q100
kept = [
    record for record in records
    if not 70 <= int(record["id"][1:]) <= 100
]

# Write to a temporary file, then replace the original
temp = path.with_suffix(".jsonl.tmp")
with temp.open("w", encoding="utf-8") as f:
    for record in kept:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

temp.replace(path)

print(f"Original records: {len(records)}")
print(f"Remaining records: {len(kept)}")
print(f"Removed records: {len(records) - len(kept)}")
print(f"Backup: {backup}")