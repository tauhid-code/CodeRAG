# CodeRAG — Retrieval-Augmented Generation for Python Codebases

CodeRAG is a code-focused Retrieval-Augmented Generation (RAG) system designed to answer questions about Python repositories using their actual source code. It ingests Python files, splits them into code-aware chunks, indexes the chunks for retrieval, reranks relevant results, and generates answers grounded in the retrieved context.

The project combines semantic search, lexical search, hybrid retrieval, and cross-encoder reranking to improve the relevance of code retrieved for a developer's question.

## Key Features

- **GitHub repository ingestion:** Load Python source files from a cloned repository.
- **Code-aware chunking:** Split Python files into structured chunks containing code symbols and line information.
- **Semantic retrieval:** Use `BAAI/bge-small-en-v1.5` embeddings with FAISS.
- **Lexical retrieval:** Use BM25 to retrieve code through keyword and identifier matches.
- **Hybrid retrieval:** Combine FAISS and BM25 rankings using Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking:** Rerank retrieved candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Query expansion:** Add code-oriented terms for questions about functions, classes, files, configuration, and models.
- **LLM-based answer generation:** Generate answers using retrieved code context through the configured LLM client.
- **Source attribution:** Return source file paths, symbols, line ranges, and retrieved code chunks alongside answers.
- **RAGAS evaluation:** Support evaluation of generated answers and retrieval quality using RAGAS.
- **Manual testing:** Test the complete pipeline against a real Python repository.

## Architecture

```text
Public GitHub Repository
          |
          v
   Python File Loader
          |
          v
    Code Chunking
          |
          v
     Embeddings
          |
          v
   FAISS Vector Index
          |
          +--------------------+
          |                    |
          v                    v
    Semantic Search        BM25 Search
          |                    |
          +---------+----------+
                    |
                    v
          Hybrid Retrieval (RRF)
                    |
                    v
          Cross-Encoder Reranker
                    |
                    v
          Retrieved Code Context
                    |
                    v
             LLM Generation
                    |
                    v
       Answer + Retrieved Sources
                    |
                    v
             RAGAS Evaluation
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Package and environment management | `uv` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector search | FAISS |
| Lexical search | BM25 |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | Configured LLM client |
| Evaluation | RAGAS |
| Version control | Git and GitHub |

## Project Structure

```text
CodeRAG/
├── app/
│   ├── embeddings/
│   │   └── embedding_model.py
│   ├── generation/
│   │   ├── groq_client.py
│   │   └── llm_client.py
│   ├── ingestion/
│   │   ├── github_loader.py
│   │   ├── code_chunker.py
│   │   └── repository_indexer.py
│   ├── retrieval/
│   │   ├── faiss_store.py
│   │   ├── bm25_store.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   └── pipeline/
│       └── rag_pipeline.py
├── data/
│   └── evaluation/
├── scripts/
│   └── manual_rag_test.py
├── tests/
├── UI/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

*The structure above highlights the main components; exact filenames may vary with the current implementation.*

## How It Works

### 1. Repository Ingestion

The repository loader discovers eligible Python files in a local repository. Unnecessary directories, such as virtual environments and Git metadata, are excluded.

### 2. Code Chunking and Indexing

The chunker extracts code-oriented chunks. The embedding model converts these chunks into vectors, which are stored in a FAISS index. Chunk metadata is saved separately so retrieved vectors can be mapped back to their source code.

### 3. Hybrid Retrieval

CodeRAG combines two retrieval approaches:

- **FAISS:** Finds semantically similar code.
- **BM25:** Finds code matching relevant keywords, identifiers, and terms.

The results are combined through Reciprocal Rank Fusion with `rrf_k=60`.

### 4. Reranking

A cross-encoder reranks candidate chunks against the original question. The highest-ranked chunks are selected as context for answer generation.

### 5. Answer Generation

The configured LLM receives the original question and retrieved code context. CodeRAG returns the generated answer together with source metadata, including file paths, symbols, and line ranges.

### 6. Evaluation

RAGAS can be used to evaluate RAG performance using an appropriate evaluation dataset and metrics. Evaluation results should be reported with the dataset, metric configuration, and actual measured scores.

## Getting Started

### Prerequisites

- Python version compatible with `.python-version` and `pyproject.toml`
- Git
- `uv`
- An API key for the configured generation provider

### 1. Clone the Project

```bash
git clone https://github.com/tauhid-code/CodeRAG.git
cd CodeRAG
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure the LLM

Configure the API key and model settings required by the LLM client in your implementation.

Keep API keys in environment variables or a local `.env` file. Never commit API keys or other secrets to GitHub.

### 4. Load a Python Repository

Clone or otherwise make a public Python repository available locally. For example:

```bash
git clone https://github.com/tauhid-code/ML-End-To-End-Project.git data/repositories/ml_end_to_end_test
```

This example assumes that the target directory does not already exist.

### 5. Index the Repository and Ask Questions

Use the repository-indexing and question-answering entry points provided by the project. The pipeline must be initialized, the repository indexed, and the LLM configured before questions can be answered.

For the current manual smoke test, see:

```text
scripts/manual_rag_test.py
```

Run the script with:

```bash
uv run python scripts/manual_rag_test.py
```

This command applies only if the script is configured to use the repository and index paths you want to test.

## Evaluation

The project includes work on both manual pipeline testing and RAGAS-based evaluation.

Manual smoke tests check whether CodeRAG can:

- Index a Python repository.
- Retrieve relevant functions, classes, and source files.
- Generate answers grounded in retrieved code.
- Return source metadata for inspection.

RAGAS evaluation provides a more systematic way to measure performance. Depending on the metrics and dataset configured, evaluation can examine answer quality, faithfulness to retrieved context, and retrieval relevance.

**Evaluation reporting:** Publish actual RAGAS scores, dataset size, metric names, and evaluation settings only after confirming them from a completed evaluation run. Successful manual tests alone do not establish a quantitative RAGAS score.

## Current Validation

The pipeline has been manually tested against the public `ML-End-To-End-Project` repository.

The recorded indexing run reported:

- Python files found: **13**
- Chunks indexed: **31**
- Vectors stored: **31**
- Failed files: **0**

The manual test also produced an answer with retrieved source information, including source file paths, symbols, line ranges, and reranking scores.

These results demonstrate a successful indexing run and a working manual question-answering path for that test. They are not a substitute for broader retrieval testing or quantitative evaluation.

## Future Improvements

- Improve retrieval quality for questions involving configuration values and model names.
- Expand test coverage across multiple Python repositories.
- Compare retrieval strategies using a fixed evaluation dataset.
- Report RAGAS metrics across repeatable evaluation runs.
- Improve handling of unsupported questions and insufficient retrieval context.
- Add a user-friendly interface for repository selection and question answering.

## Limitations

- Answer quality depends on the quality of the indexed code, retrieval results, and generation model.
- Poor retrieval can result in incomplete or incorrect answers even when the relevant code exists in the repository.
- Indexes must be regenerated when repository code changes.
- RAGAS scores depend on the dataset, metrics, evaluator configuration, and model settings.

## License

Choose and add an appropriate open-source license before distributing the project under a specific license.

## Author

**Tauhid Ur Rehman**

GitHub: [@tauhid-code](https://github.com/tauhid-code)

Project repository: [CodeRAG](https://github.com/tauhid-code/CodeRAG)
