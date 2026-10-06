
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_question.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results"
    / "groq_results.jsonl"
)

RESULTS_DIR = RESULTS_PATH.parent

SUMMARY_PATH = (
    RESULTS_DIR / "generation_evaluation_summary.json"
)

CSV_PATH = (
    RESULTS_DIR / "generation_evaluation_per_question.csv"
)


# ============================================================
# TEXT NORMALIZATION AND METRICS
# ============================================================

def normalize_text(text):
    """Normalize text for comparisons."""
    text = str(text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def tokenize(text):
    """Tokenize normalized text."""
    return re.findall(r"\w+", str(text or "").lower())


def calculate_f1(reference, prediction):
    """Calculate token-level precision, recall and F1."""
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    pred_counts = Counter(pred_tokens)

    overlap = sum(
        (ref_counts & pred_counts).values()
    )

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return (
        2 * precision * recall
        / (precision + recall)
    )


def calculate_rouge1_f1(reference, prediction):
    """Calculate ROUGE-1 F1 using unigram overlap."""
    return calculate_f1(reference, prediction)


def calculate_expected_point_coverage(answer, expected_points):
    """Calculate the fraction of expected points found in an answer."""
    if not expected_points:
        return None

    normalized_answer = normalize_text(answer)

    if not normalized_answer:
        return 0.0

    answer_tokens = normalized_answer.split()
    matched = 0

    for point in expected_points:
        point_tokens = normalize_text(point).split()

        if not point_tokens:
            continue

        found = any(
            answer_tokens[i:i + len(point_tokens)] == point_tokens
            for i in range(
                len(answer_tokens) - len(point_tokens) + 1
            )
        )

        if found:
            matched += 1

    return matched / len(expected_points)


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(record):
    """
    Support the actual nested Groq format, previous Gemini
    format and simple top-level answer formats.
    """
    if not isinstance(record, dict):
        return ""

    # Current Groq format
    groq = record.get("groq")
    if isinstance(groq, dict):
        answer = groq.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    # Previous Gemini format
    gemini = record.get("gemini")
    if isinstance(gemini, dict):
        answer = gemini.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    # Other possible generation formats
    for key in ("answer", "generated_answer", "response"):
        answer = record.get(key)
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    # Generic generation field
    generation = record.get("generation")
    if isinstance(generation, dict):
        answer = generation.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

    return ""


def extract_latency(record):
    """Get generation latency from a nested or top-level field."""
    if not isinstance(record, dict):
        return None

    candidates = []

    for key in ("groq", "gemini", "generation"):
        section = record.get(key)
        if isinstance(section, dict):
            candidates.append(section.get("latency_seconds"))

    candidates.append(record.get("latency_seconds"))

    for value in candidates:
        try:
            if value is not None:
                latency = float(value)
                if latency >= 0:
                    return latency
        except (TypeError, ValueError):
            continue

    return None


# ============================================================
# ABSTENTION DETECTION
# ============================================================

ABSTENTION_PATTERNS = [
    r"\bcannot determine\b",
    r"\bcan't determine\b",
    r"\bcannot confirm\b",
    r"\bcan't confirm\b",
    r"\bcannot verify\b",
    r"\bcan't verify\b",
    r"\bunable to verify\b",
    r"\bunable to determine\b",
    r"\bunable to confirm\b",
    r"\binsufficient information\b",
    r"\bnot enough information\b",
    r"\bnot available\b",
    r"\bunavailable\b",
    r"\bnot provided\b",
    r"\bnot present in the (?:provided )?(?:source|context|code)\b",
    r"\bnot included in the (?:provided )?(?:source|context)\b",
    r"\bnot specified\b",
    r"\bnot shown\b",
    r"\bsource file is missing\b",
    r"\bsource files? (?:is|are) unavailable\b",
    r"\bmissing source\b",
    r"\boutside the (?:provided )?context\b",
    r"\bdoes not contain enough information\b",
    r"\bno information (?:is|was) available\b",
    r"\bi don't know\b",
    r"\bi do not know\b",
    r"\bi cannot answer\b",
    r"\bi can't answer\b",
]


def is_abstention(answer):
    """
    Heuristic only: detect explicit statements that the
    answer cannot be established from available information.
    """
    if not answer or not answer.strip():
        return False

    text = answer.lower()

    return any(
        re.search(pattern, text)
        for pattern in ABSTENTION_PATTERNS
    )


# ============================================================
# LOAD DATASET AND RESULTS
# ============================================================

def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON list.")

    seen = set()

    for item in dataset:
        if not isinstance(item, dict):
            raise ValueError("Every dataset entry must be an object.")

        for field in (
            "id",
            "question",
            "answerable",
            "reference_answer",
            "expected_points",
            "source_files",
        ):
            if field not in item:
                raise ValueError(
                    f"Missing field {field!r} in dataset record."
                )

        question_id = item["id"]

        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("Invalid question ID in dataset.")

        if question_id in seen:
            raise ValueError(
                f"Duplicate question ID in dataset: {question_id}"
            )

        seen.add(question_id)

        if not isinstance(item["answerable"], bool):
            raise ValueError(
                f"Invalid answerable value for {question_id}"
            )

        if not isinstance(item["expected_points"], list):
            raise ValueError(
                f"expected_points must be a list for {question_id}"
            )

    print(f"Loaded dataset questions: {len(dataset)}")
    return dataset


def load_results(path):
    """
    Load JSONL records.

    For duplicate IDs, prefer the last record with a non-empty
    answer. If no record has an answer, use the last record.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    records = []
    selected = {}
    duplicate_ids = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} "
                    f"of {path}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record on line {line_number} must be an object."
                )

            question_id = record.get("id")

            if not isinstance(question_id, str) or not question_id:
                raise ValueError(
                    f"Missing or invalid ID on line {line_number}."
                )

            records.append(record)

            if question_id in selected:
                duplicate_ids.add(question_id)

            previous = selected.get(question_id)

            if previous is None:
                selected[question_id] = record
            else:
                previous_answer = extract_answer(previous)
                current_answer = extract_answer(record)

                if current_answer or not previous_answer:
                    selected[question_id] = record

    print(f"Result records: {len(records)}")
    print(f"Unique result IDs: {len(selected)}")
    print(f"Duplicate IDs: {len(duplicate_ids)}")

    return records, selected, sorted(duplicate_ids)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(dataset, results):
    details = []

    token_f1_scores = []
    rouge_scores = []
    point_coverages = []
    latencies = []

    answerable_count = 0
    unanswerable_count = 0

    successful = 0
    missing_or_failed = 0

    correctly_answered = 0
    correctly_abstained = 0
    incorrectly_answered = 0
    incorrectly_abstained = 0

    answerable_responses = 0
    unanswerable_abstentions = 0

    missing_ids = []

    for item in dataset:
        question_id = item["id"]
        answerable = item["answerable"]

        if answerable:
            answerable_count += 1
        else:
            unanswerable_count += 1

        record = results.get(question_id)

        if record is None:
            answer = ""
            status = "missing"
            latency = None
        else:
            answer = extract_answer(record)
            latency = extract_latency(record)

            # An actual non-empty answer is considered generated,
            # even if the older record has no status field.
            status_field = record.get("status")

            if answer:
                status = "success"
            elif status_field == "failed":
                status = "failed"
            else:
                status = "missing_answer"

        generated = bool(answer.strip())

        if generated:
            successful += 1
            if latency is not None:
                latencies.append(latency)
        else:
            missing_or_failed += 1
            missing_ids.append(question_id)

        abstained = is_abstention(answer) if generated else False

        token_f1 = None
        rouge1_f1 = None
        point_coverage = None

        if answerable and generated:
            reference = item["reference_answer"]

            token_f1 = calculate_f1(reference, answer)
            rouge1_f1 = calculate_rouge1_f1(
                reference, answer
            )
            point_coverage = calculate_expected_point_coverage(
                answer, item["expected_points"]
            )

            token_f1_scores.append(token_f1)
            rouge_scores.append(rouge1_f1)

            if point_coverage is not None:
                point_coverages.append(point_coverage)

            if abstained:
                incorrectly_abstained += 1
            else:
                correctly_answered += 1
                answerable_responses += 1

        elif not answerable and generated:
            if abstained:
                correctly_abstained += 1
                unanswerable_abstentions += 1
            else:
                incorrectly_answered += 1

        details.append({
            "id": question_id,
            "category": item.get("category"),
            "question": item["question"],
            "answerable": answerable,
            "generation_status": status,
            "answer": answer,
            "reference_answer": item["reference_answer"],
            "token_f1": token_f1,
            "rouge1_f1": rouge1_f1,
            "expected_point_coverage": point_coverage,
            "expected_points_count": len(item["expected_points"]),
            "abstention_detected": abstained if generated else None,
            "latency_seconds": latency,
            "source_files": json.dumps(
                item["source_files"], ensure_ascii=False
            ),
        })

    total = len(dataset)

    # Failed or missing generations are incorrect for the
    # overall answerability heuristic accuracy.
    accuracy = (
        (correctly_answered + correctly_abstained) / total
        if total else None
    )

    answerable_response_rate = (
        answerable_responses / answerable_count
        if answerable_count else None
    )

    unanswerable_abstention_rate = (
        unanswerable_abstentions / unanswerable_count
        if unanswerable_count else None
    )

    summary = {
        "dataset_questions": total,
        "result_records": None,  # Filled in by main()
        "unique_result_ids": len(results),
        "successful_generations": successful,
        "missing_or_failed_generations": missing_or_failed,
        "success_rate": successful / total if total else None,

        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,

        "answerable_answer_token_f1": (
            mean(token_f1_scores) if token_f1_scores else None
        ),
        "answerable_answer_rouge1_f1": (
            mean(rouge_scores) if rouge_scores else None
        ),
        "expected_point_coverage": (
            mean(point_coverages) if point_coverages else None
        ),

        "answerability_heuristic_accuracy": accuracy,
        "answerable_response_rate": answerable_response_rate,
        "unanswerable_abstention_rate": (
            unanswerable_abstention_rate
        ),

        "answerability_confusion_counts": {
            "correctly_answered_answerable": correctly_answered,
            "correctly_abstained_unanswerable": correctly_abstained,
            "incorrectly_answered_unanswerable": incorrectly_answered,
            "incorrectly_abstained_answerable": incorrectly_abstained,
        },

        "average_latency_seconds": (
            mean(latencies) if latencies else None
        ),
        "median_latency_seconds": (
            median(latencies) if latencies else None
        ),

        "duplicate_result_ids": [],
        "missing_question_ids": missing_ids,
    }

    return summary, details


# ============================================================
# SAVE REPORTS
# ============================================================

def save_summary(summary, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_csv(details, path):
    if not details:
        print("No per-question details to save.")
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(details[0].keys()),
        )
        writer.writeheader()
        writer.writerows(details)


def print_summary(summary):
    print("\n" + "=" * 65)
    print("GENERATION EVALUATION")
    print("=" * 65)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("=" * 65)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved CodeRAG generation results."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the generation question dataset.",
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the generated JSONL results.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to save the JSON summary.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to save per-question CSV details.",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    raw_records, results, duplicate_ids = load_results(
        args.results
    )

    summary, details = evaluate(dataset, results)

    summary["result_records"] = len(raw_records)
    summary["duplicate_result_ids"] = duplicate_ids

    args.summary.parent.mkdir(
        parents=True, exist_ok=True
    )
    args.csv.parent.mkdir(
        parents=True, exist_ok=True
    )

    save_summary(summary, args.summary)
    save_csv(details, args.csv)

    print_summary(summary)

    print(f"\nSaved summary: {args.summary}")
    print(f"Saved per-question report: {args.csv}")
    print(f"Results source: {args.results}")


if __name__ == "__main__":
    main()
