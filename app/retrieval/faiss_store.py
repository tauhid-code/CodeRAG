from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from app.ingestion.code_chunker import CodeChunk


class FAISSVectorStore:

    def __init__(self, dimension: int = 384):

        self.dimension = dimension

        # Inner Product works well because
        # our BGE embeddings are normalized.
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.chunks: List[CodeChunk] = []

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        chunks: List[CodeChunk]
    ):

        if not embeddings:
            print("No embeddings to add.")
            return

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings and chunks "
                "must be the same."
            )

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        self.index.add(vectors)

        self.chunks.extend(chunks)

        print(
            f"Added {len(chunks)} chunks "
            f"to FAISS."
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[CodeChunk, float]]:

        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append(
                (chunk, float(score))
            )

        return results

    def save(
        self,
        index_path: str,
        metadata_path: str
    ):

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(index_path)
        )

        import pickle

        with open(
            metadata_path,
            "wb"
        ) as file:

            pickle.dump(
                self.chunks,
                file
            )

        print(
            f"FAISS index saved to: "
            f"{index_path}"
        )

        print(
            f"Metadata saved to: "
            f"{metadata_path}"
        )

    def load(
        self,
        index_path: str,
        metadata_path: str
    ):

        import pickle

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "rb"
        ) as file:

            self.chunks = pickle.load(
                file
            )

        print(
            f"Loaded {self.index.ntotal} "
            f"vectors from FAISS."
        )