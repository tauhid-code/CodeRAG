import argparse

import json

import os

import re

import statistics

from pathlib import Path

from typing import Any

from dotenv import load_dotenv

from datasets import Dataset

from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings



from ragas import evaluate

from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.llms import LangchainLLMWrapper

from ragas.run_config import RunConfig

from ragas.metrics import (

    AnswerCorrectness,

    AnswerRelevancy,

    ContextPrecision,

    ContextRecall,

    Faithfulness,



)







ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUESTIONS = (



    ROOT

    / "data"

    / "evaluation"

    / "generation_question.json"



)



DEFAULT_RESULTS = (

    ROOT

    / "data"

    / "evaluation"

    / "results"

    / "groq_results.jsonl"

)

OUTPUT_DIR = (

    ROOT

    / "tests"

    / "ragas_results"

)



# Same embedding model used by CodeRAG.

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"



# This is the Groq-served model.

# ChatOpenAI is only being used as the OpenAI-compatible

# client interface. Requests are sent to Groq through base_url.

DEFAULT_EVALUATOR_MODEL = "openai/gpt-oss-20b"



def load_json(path: Path) -> Any:



    """Load a normal JSON file."""



    with path.open("r", encoding="utf-8") as file:



        return json.load(file)











def load_results(path: Path) -> list[dict]:



    """



    Load saved generation results.







    Supports:



    - JSONL



    - JSON list



    - JSON object containing results/records/data



    - Single JSON record



    """







    if not path.exists():



        raise FileNotFoundError(



            f"Results file not found: {path}"



        )







    if path.suffix.lower() == ".jsonl":







        records = []







        with path.open("r", encoding="utf-8") as file:







            for line_number, line in enumerate(file, start=1):







                if not line.strip():



                    continue







                try:



                    records.append(json.loads(line))







                except json.JSONDecodeError as exc:







                    raise ValueError(



                        f"Invalid JSON on line {line_number} "



                        f"of {path}"



                    ) from exc







        return records







    # -----------------------------------------------------



    # Normal JSON



    # -----------------------------------------------------







    data = load_json(path)







    if isinstance(data, list):



        return data







    if isinstance(data, dict):







        for key in (



            "results",



            "records",



            "data",



        ):







            if isinstance(data.get(key), list):



                return data[key]







        if "id" in data:



            return [data]







    raise ValueError(



        f"Unsupported result format in {path}. "



        "Expected JSONL, JSON list, or a supported JSON object."



    )











# =========================================================



# ANSWER EXTRACTION



# =========================================================







def extract_answer(record: dict) -> str:



    """



    Extract generated answer from the saved result.







    Current CodeRAG format:







        {



            "groq": {



                "answer": "..."



            }



        }







    Also supports older Gemini/flat formats.



    """







    # Current Groq format.



    for provider in (



        "groq",



        "gemini",



    ):







        value = record.get(provider)







        if isinstance(value, dict):







            answer = value.get("answer")







            if isinstance(answer, str):



                return answer.strip()







    # Fallback formats.



    for key in (



        "generated_answer",



        "answer",



        "response",



    ):







        value = record.get(key)







        if isinstance(value, str):



            return value.strip()







    return ""











# =========================================================



# CONTEXT EXTRACTION



# =========================================================







def extract_contexts(record: dict) -> list[str]:



    """



    Extract retrieved code chunks from the saved retrieval object.







    Current CodeRAG format:







        record["retrieval"]["chunks"]







    Each chunk contains:







        content







    Other common field names are also supported.



    """







    retrieval = record.get(



        "retrieval",



        {}



    )







    if not isinstance(retrieval, dict):



        return []







    chunks = retrieval.get(



        "chunks",



        []



    )







    if not isinstance(chunks, list):



        return []







    contexts = []







    for chunk in chunks:







        # -------------------------------------------------



        # If the chunk itself is a string.



        # -------------------------------------------------







        if isinstance(chunk, str):







            content = chunk







        # -------------------------------------------------



        # If the chunk is a dictionary.



        # -------------------------------------------------







        elif isinstance(chunk, dict):







            content = None







            for key in (



                "content",



                "page_content",



                "text",



                "code",



                "chunk",



            ):







                value = chunk.get(key)







                if isinstance(value, str):







                    content = value



                    break







        else:







            content = None







        if (



            isinstance(content, str)



            and content.strip()



        ):







            contexts.append(



                content.strip()



            )







    return contexts











# =========================================================



# ABSTENTION DETECTION



# =========================================================







ABSTENTION_PATTERNS = [







    r"\bI (?:cannot|can't|couldn't|do not|don't) "



    r"(?:find|answer|determine|provide|verify)\b",







    r"\bnot (?:available|present|provided|found) "



    r"in (?:the )?(?:repository|context|source|code)\b",







    r"\bthere is insufficient information\b",







    r"\bthe (?:repository|provided context|available code) "



    r"does not\b",







    r"\bI don't have enough information\b",







    r"\bI am unable to\b",



]











def is_abstention(answer: str) -> bool:



    """



    Simple heuristic for unanswerable questions.







    This is NOT a RAGAS metric.



    """







    if not answer.strip():



        return False







    for pattern in ABSTENTION_PATTERNS:







        if re.search(



            pattern,



            answer,



            flags=re.IGNORECASE,



        ):



            return True







    return False











# =========================================================



# BUILD EVALUATION RECORDS



# =========================================================







def build_records(



    questions: list[dict],



    results: list[dict],



) -> tuple[list[dict], list[dict]]:







    """



    Match the evaluation questions with saved generation results.



    """







    result_by_id = {}







    for result in results:







        question_id = result.get("id")







        if question_id is not None:







            result_by_id[



                str(question_id)



            ] = result







    answerable_records = []



    unanswerable_records = []







    for question in questions:







        question_id = str(



            question.get("id", "")



        )







        result = result_by_id.get(



            question_id



        )







        # -------------------------------------------------



        # No generation result.



        # -------------------------------------------------







        if result is None:







            answer = ""



            contexts = []







        else:







            answer = extract_answer(



                result



            )







            contexts = extract_contexts(



                result



            )







            # Fallback in case contexts were stored



            # inside the question record.



            if not contexts:







                contexts = extract_contexts(



                    question



                )







        record = {







            "id": question_id,







            "category": question.get(



                "category",



                "",



            ),







            "question": question.get(



                "question",



                "",



            ),







            "reference": question.get(



                "reference_answer",



                "",



            ),







            "answer": answer,







            "contexts": contexts,







            "answerable": bool(



                question.get(



                    "answerable",



                    False,



                )



            ),







            "has_result": result is not None,



        }







        if record["answerable"]:







            answerable_records.append(



                record



            )







        else:







            unanswerable_records.append(



                record



            )







    return (



        answerable_records,



        unanswerable_records,



    )











# =========================================================



# SAVE JSON



# =========================================================







def write_json(



    path: Path,



    data: Any,



) -> None:







    path.parent.mkdir(



        parents=True,



        exist_ok=True,



    )







    with path.open(



        "w",



        encoding="utf-8",



    ) as file:







        json.dump(



            data,



            file,



            indent=2,



            ensure_ascii=False,



        )











# =========================================================



# RAGAS EVALUATION



# =========================================================







def run_ragas(



    records: list[dict],



    evaluator_model: str,



) -> tuple[list[dict], dict]:







    """



    Run RAGAS on answerable questions that have:







    1. Generated answer



    2. Reference answer



    3. Retrieved contexts



    """







    # -----------------------------------------------------



    # Select usable records.



    # -----------------------------------------------------







    usable = [







        record







        for record in records







        if (



            record["answer"].strip()



            and record["reference"].strip()



            and record["contexts"]



        )







    ]







    if not usable:







        raise ValueError(



            "No answerable records contain all required "



            "fields: answer, reference answer and contexts."



        )







    print(



        f"\nUsable RAGAS records: {len(usable)}"



    )







    # -----------------------------------------------------



    # Convert to RAGAS dataset format.



    # -----------------------------------------------------







    dataset = Dataset.from_list(







        [







            {



                "user_input": record["question"],



                "response": record["answer"],



                "retrieved_contexts": record["contexts"],



                "reference": record["reference"],



            }







            for record in usable







        ]







    )







    # =====================================================



    # GROQ JUDGE



    # =====================================================







    # ChatOpenAI is only the client interface.



    #



    # base_url points to Groq.



    #



    # Therefore:



    #



    #     NO OPENAI API KEY IS USED.



    #



    # The existing GROQ_API_KEY is used.



    # =====================================================







    judge = ChatOpenAI(







        model=evaluator_model,







        base_url="https://api.groq.com/openai/v1",







        api_key=os.environ[



            "GROQ_API_KEY"



        ],







        temperature=0,







        # Increased from 2048 because the



        # previous run produced:



        #



        # LLMDidNotFinishException



        #



        max_tokens=4096,







        # Give Groq enough time to respond.



        timeout=180,







        # LangChain-level retries.



        max_retries=3,



    )







    # =====================================================



    # EMBEDDINGS



    # =====================================================







    embeddings = HuggingFaceEmbeddings(







        model_name=EMBEDDING_MODEL,







        model_kwargs={



            "device": "cpu"



        },







        encode_kwargs={



            "normalize_embeddings": True



        },



    )







    # =====================================================



    # RAGAS WRAPPERS



    # =====================================================







    ragas_llm = LangchainLLMWrapper(



        judge



    )







    ragas_embeddings = (



        LangchainEmbeddingsWrapper(



            embeddings



        )



    )







    # =====================================================



    # METRICS



    # =====================================================







    metrics = [







        Faithfulness(),







        AnswerRelevancy(),







        AnswerCorrectness(),







        ContextPrecision(),







        ContextRecall(),







    ]







    print(



        f"\nEvaluating {len(usable)} "



        "answerable records with RAGAS..."



    )







    print(



        f"Evaluator model: {evaluator_model}"



    )







    print(



        "Evaluator provider: Groq"



    )







    print(



        "Judge output limit: 4096 tokens"



    )







    print(



        "RAGAS workers: 1"



    )







    print(



        "This may take some time and use API quota.\n"



    )







    # =====================================================



    # RAGAS RUN CONFIG



    # =====================================================







    # IMPORTANT:



    #



    # The previous version imported RunConfig



    # but did NOT pass it to evaluate().



    #



    # This configuration limits concurrent requests



    # and gives failed requests additional retry time.



    # =====================================================







    run_config = RunConfig(







        # Maximum time allowed for an individual



        # evaluation operation.



        timeout=180,







        # Retry failed operations.



        max_retries=3,







        # Maximum wait between retries.



        max_wait=30,







        # IMPORTANT:



        # Use one worker to reduce simultaneous



        # Groq requests and rate-limit pressure.



        max_workers=1,



    )







    # =====================================================



    # RUN RAGAS



    # =====================================================







    result = evaluate(







        dataset=dataset,







        metrics=metrics,







        llm=ragas_llm,







        embeddings=ragas_embeddings,







        run_config=run_config,







        # Don't stop the entire evaluation because



        # one individual metric/question fails.



        raise_exceptions=False,



    )







    # -----------------------------------------------------



    # Convert results to pandas.



    # -----------------------------------------------------







    df = result.to_pandas()







    # -----------------------------------------------------



    # Expected metric names.



    # -----------------------------------------------------







    metric_columns = [







        "faithfulness",







        "answer_relevancy",







        "answer_correctness",







        "context_precision",







        "context_recall",







    ]







    # -----------------------------------------------------



    # Build per-question output.



    # -----------------------------------------------------







    per_question = []







    for index, record in enumerate(



        usable



    ):







        row = {







            "id": record["id"],







            "category": record[



                "category"



            ],







            "question": record[



                "question"



            ],







            "answer": record[



                "answer"



            ],







            "reference_answer": record[



                "reference"



            ],







            "context_count": len(



                record["contexts"]



            ),



        }







        for metric in metric_columns:







            if metric in df.columns:







                value = df.iloc[index][



                    metric



                ]







                # Pandas may represent missing



                # values as NaN.



                if (



                    value is None



                    or str(value) == "nan"



                ):







                    row[metric] = None







                else:







                    row[metric] = float(



                        value



                    )







            else:







                row[metric] = None







        per_question.append(



            row



        )







    # =====================================================



    # SUMMARY



    # =====================================================







    summary = {}







    for metric in metric_columns:







        values = [







            row[metric]







            for row in per_question







            if isinstance(



                row.get(metric),



                (int, float),



            )







        ]







        if values:







            summary[metric] = round(



                statistics.mean(values),



                4,



            )







        else:







            summary[metric] = None







    summary[



        "evaluated_count"



    ] = len(per_question)







    summary[



        "not_evaluated_count"



    ] = (



        len(records)



        - len(per_question)



    )







    return (



        per_question,



        summary,



    )











# =========================================================



# MAIN



# =========================================================







def main():







    parser = argparse.ArgumentParser(







        description=(



            "Evaluate saved CodeRAG answers "



            "using RAGAS."



        )



    )







    # -----------------------------------------------------



    # Questions file.



    # -----------------------------------------------------







    parser.add_argument(







        "--questions",







        type=Path,







        default=DEFAULT_QUESTIONS,







        help=(



            "Path to the evaluation "



            "question JSON file."



        ),



    )







    # -----------------------------------------------------



    # Results file.



    # -----------------------------------------------------







    parser.add_argument(







        "--results",







        type=Path,







        default=DEFAULT_RESULTS,







        help=(



            "Path to the saved Groq "



            "results JSON/JSONL file."



        ),



    )







    # -----------------------------------------------------



    # Evaluator model.



    # -----------------------------------------------------







    parser.add_argument(







        "--model",







        default=None,







        help=(



            "Groq model used as the "



            "RAGAS evaluation judge."



        ),



    )



    # -----------------------------------------------------

    # Limit / batch range.

    # -----------------------------------------------------



    parser.add_argument(

        "--limit",

        type=int,

        default=0,

        help=(

            "Evaluate only the first N answerable questions. "

            "0 means all. Do not use together with --start/--end."

        ),

    )



    parser.add_argument(

        "--start",

        type=int,

        default=0,

        help=(

            "1-based starting position among answerable questions. "

            "Example: --start 6 --end 10."

        ),

    )



    parser.add_argument(

        "--end",

        type=int,

        default=0,

        help=(

            "1-based inclusive ending position among answerable "

            "questions."

        ),

    )



    args = parser.parse_args()







    # =====================================================



    # LOAD ENVIRONMENT



    # =====================================================







    load_dotenv(



        ROOT / ".env"



    )







    # -----------------------------------------------------



    # API key check.



    # -----------------------------------------------------







    groq_api_key = os.getenv(



        "GROQ_API_KEY"



    )







    if not groq_api_key:







        raise EnvironmentError(



            "GROQ_API_KEY is missing. "



            "Add it to your .env file."



        )







    # -----------------------------------------------------



    # Resolve evaluator model AFTER



    # loading .env.



    # -----------------------------------------------------







    evaluator_model = (



        args.model



        or os.getenv(



            "RAGAS_EVALUATOR_MODEL",



            DEFAULT_EVALUATOR_MODEL,



        )



    )







    # =====================================================



    # VALIDATE INPUT FILES



    # =====================================================







    if not args.questions.exists():







        raise FileNotFoundError(







            "Question file not found:\n"



            f"{args.questions}"



        )







    if not args.results.exists():







        raise FileNotFoundError(







            "Results file not found:\n"



            f"{args.results}"



        )







    # =====================================================



    # LOAD DATA



    # =====================================================







    questions = load_json(



        args.questions



    )







    if not isinstance(



        questions,



        list,



    ):







        raise ValueError(



            "Question file must contain "



            "a JSON array."



        )







    results = load_results(



        args.results



    )







    # =====================================================



    # BUILD RECORDS



    # =====================================================







    (



        answerable,



        unanswerable,



    ) = build_records(



        questions,



        results,



    )



    # =====================================================

    # SELECT EVALUATION BATCH

    # =====================================================



    if args.limit > 0 and (args.start > 0 or args.end > 0):

        raise ValueError(

            "Use either --limit OR --start/--end, not both."

        )



    if args.start < 0 or args.end < 0:

        raise ValueError(

            "--start and --end must be positive integers."

        )



    if args.start > 0 or args.end > 0:

        if args.start <= 0 or args.end <= 0:

            raise ValueError(

                "For batch evaluation, provide both --start and --end."

            )

        if args.start > args.end:

            raise ValueError(

                "--start cannot be greater than --end."

            )

        if args.end > len(answerable):

            raise ValueError(

                f"Batch end {args.end} exceeds the number of "

                f"answerable questions ({len(answerable)})."

            )



        batch_start = args.start

        batch_end = args.end

        answerable = answerable[batch_start - 1:batch_end]

        batch_suffix = f"_Q{batch_start:03d}_Q{batch_end:03d}"



    elif args.limit > 0:

        answerable = answerable[:args.limit]

        batch_suffix = f"_Q001_Q{args.limit:03d}"

    else:

        batch_suffix = "_ALL"







    # =====================================================



    # BASIC INFORMATION



    # =====================================================







    print(



        "\n" + "=" * 60



    )







    print(



        "CodeRAG - RAGAS Evaluation"



    )







    print(



        "=" * 60



    )







    print(



        f"Questions in dataset: "



        f"{len(questions)}"



    )







    print(



        "Answerable questions: "



        f"{sum(bool(q.get('answerable', False)) for q in questions)}"



    )







    print(



        f"Unanswerable questions: "



        f"{len(unanswerable)}"



    )







    print(



        f"Saved result records: "



        f"{len(results)}"



    )







    print(



        f"Evaluator model: "



        f"{evaluator_model}"



    )







    # =====================================================



    # RUN RAGAS



    # =====================================================







    evaluated, summary = run_ragas(



        answerable,



        evaluator_model,



    )







    # =====================================================



    # UNANSWERABLE QUESTIONS



    # =====================================================







    # These are NOT evaluated using the reference-based



    # RAGAS metrics above.



    #



    # Instead, we check whether the generated answer



    # appears to abstain using our simple heuristic.



    # =====================================================







    abstention_rows = []







    for record in unanswerable:







        abstention_rows.append({







            "id": record["id"],







            "category": record[



                "category"



            ],







            "has_result": record[



                "has_result"



            ],







            "abstained": is_abstention(



                record["answer"]



            ),







            "answer": record[



                "answer"



            ],



        })







    attempted = [







        row







        for row in abstention_rows







        if row["has_result"]







    ]







    abstention_count = sum(







        row["abstained"]







        for row in attempted







    )







    summary[



        "unanswerable_total"



    ] = len(



        unanswerable



    )







    summary[



        "unanswerable_with_result"



    ] = len(



        attempted



    )







    summary[



        "unanswerable_abstention_count"



    ] = abstention_count







    if attempted:







        summary[



            "unanswerable_abstention_rate"



        ] = round(







            abstention_count



            / len(attempted),







            4,



        )







    else:







        summary[



            "unanswerable_abstention_rate"



        ] = None







    # =====================================================



    # SAVE OUTPUTS



    # =====================================================







    OUTPUT_DIR.mkdir(



        parents=True,



        exist_ok=True,



    )







    # -----------------------------------------------------



    # Summary.



    # -----------------------------------------------------







    write_json(







        OUTPUT_DIR



        / f"ragas_summary{batch_suffix}.json",







        summary,



    )







    # -----------------------------------------------------



    # Per-question RAGAS results.



    # -----------------------------------------------------







    write_json(







        OUTPUT_DIR



        / f"ragas_per_question{batch_suffix}.json",







        evaluated,



    )







    # -----------------------------------------------------



    # Unanswerable results.



    # -----------------------------------------------------







    write_json(







        OUTPUT_DIR



        / f"unanswerable_abstention{batch_suffix}.json",







        abstention_rows,



    )







    # -----------------------------------------------------



    # CSV.



    # -----------------------------------------------------







    import pandas as pd







    pd.DataFrame(



        evaluated



    ).to_csv(







        OUTPUT_DIR



        / f"ragas_per_question{batch_suffix}.csv",







        index=False,







        encoding="utf-8-sig",



    )







    # =====================================================



    # PRINT FINAL SUMMARY



    # =====================================================







    print(



        "\n" + "=" * 60



    )







    print(



        "RAGAS SUMMARY"



    )







    print(



        "=" * 60



    )







    for metric, value in summary.items():







        print(



            f"{metric}: {value}"



        )







    print(



        "\nSaved evaluation files to:"



    )







    print(



        OUTPUT_DIR



    )







    print(



        "\nEvaluation completed."



    )











# =========================================================



# ENTRY POINT



# =========================================================







if __name__ == "__main__":







    main()