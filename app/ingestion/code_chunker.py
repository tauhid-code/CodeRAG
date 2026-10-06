import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CodeChunk:
    content: str
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int


class PythonCodeChunker:

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        """
        Parse one Python file and extract meaningful code chunks.

        Chunks include:
        - Module-level context/docstring/constants
        - Functions
        - Async functions
        - Classes
        """

        try:
            source_code = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            return []

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"Could not parse {file_path}: {e}")
            return []

        chunks = []

        # ---------------------------------------------------------
        # 1. Module-level context
        # ---------------------------------------------------------
        module_lines = source_code.splitlines()
        module_context_lines = []

        # Module docstring
        if tree.body:
            first_node = tree.body[0]

            if (
                isinstance(first_node, ast.Expr)
                and isinstance(first_node.value, ast.Constant)
                and isinstance(first_node.value.value, str)
            ):
                module_context_lines.extend(
                    module_lines[
                        first_node.lineno - 1:first_node.end_lineno
                    ]
                )

        # Module-level assignments/constants
        for node in tree.body:

            if isinstance(node, (ast.Assign, ast.AnnAssign)):

                module_context_lines.extend(
                    module_lines[
                        node.lineno - 1:node.end_lineno
                    ]
                )

        # Create module chunk if we found useful content
        if module_context_lines:

            chunks.append(
                CodeChunk(
                    content="\n".join(module_context_lines),
                    file_path=str(file_path),
                    symbol_name="__module__",
                    symbol_type="module",
                    start_line=1,
                    end_line=max(
                        node.end_lineno
                        for node in tree.body
                        if isinstance(
                            node,
                            (
                                ast.Expr,
                                ast.Assign,
                                ast.AnnAssign
                            )
                        )
                    )
                )
            )

        # ---------------------------------------------------------
        # 2. Functions, async functions and classes
        # ---------------------------------------------------------
        for node in ast.walk(tree):

            if isinstance(node, ast.FunctionDef):

                chunk = self._create_chunk(
                    node=node,
                    source_code=source_code,
                    file_path=file_path,
                    symbol_type="function"
                )

                chunks.append(chunk)

            elif isinstance(node, ast.AsyncFunctionDef):

                chunk = self._create_chunk(
                    node=node,
                    source_code=source_code,
                    file_path=file_path,
                    symbol_type="async_function"
                )

                chunks.append(chunk)

            elif isinstance(node, ast.ClassDef):

                chunk = self._create_chunk(
                    node=node,
                    source_code=source_code,
                    file_path=file_path,
                    symbol_type="class"
                )

                chunks.append(chunk)

        return chunks

    def chunk_repository(
        self,
        repo_path: Path,
        python_files: List[Path]
    ) -> List[CodeChunk]:
        """
        Process every Python file in the repository.
        """

        all_chunks = []

        print("\nStarting repository chunking...")
        print(f"Repository: {repo_path}")
        print(f"Python files to process: {len(python_files)}")

        for index, file_path in enumerate(
            python_files,
            start=1
        ):

            print(
                f"Processing "
                f"{index}/{len(python_files)}: "
                f"{file_path}"
            )

            file_chunks = self.chunk_file(file_path)

            all_chunks.extend(file_chunks)

        print("\nRepository chunking completed.")

        print(
            f"Total code chunks created: "
            f"{len(all_chunks)}"
        )

        return all_chunks

    def _create_chunk(
        self,
        node,
        source_code: str,
        file_path: Path,
        symbol_type: str
    ) -> CodeChunk:

        lines = source_code.splitlines()

        start_line = node.lineno
        end_line = node.end_lineno

        chunk_lines = lines[start_line - 1:end_line]

        content = "\n".join(chunk_lines)

        return CodeChunk(
            content=content,
            file_path=str(file_path),
            symbol_name=getattr(
                node,
                "name",
                "unknown"
            ),
            symbol_type=symbol_type,
            start_line=start_line,
            end_line=end_line
        )