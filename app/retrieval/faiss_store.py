from pathlib import Path
from typing import List, Tuple
import pickle

import faiss
import numpy as np

from app.ingestion.code_chunker import CodeChunk


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[CodeChunk] = []

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        chunks: List[CodeChunk],
    ):
        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings and chunks must be the same."
            )

        if not embeddings:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with shape (n, {self.dimension}), "
                f"received {vectors.shape}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain NaN or infinite values.")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[CodeChunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                "FAISS index and chunk metadata are inconsistent."
            )

        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_vector.shape != (self.dimension,):
            raise ValueError(
                f"Expected query embedding shape ({self.dimension},), "
                f"received {query_vector.shape}."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError(
                "Query embedding contains NaN or infinite values."
            )

        scores, indices = self.index.search(
            query_vector.reshape(1, -1),
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, chunk_index in zip(scores[0], indices[0]):
            if chunk_index == -1:
                continue

            results.append(
                (self.chunks[int(chunk_index)], float(score))
            )

        return results

    def save(
        self,
        index_path: str,
        metadata_path: str,
    ):
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                "Cannot save: FAISS index and chunk metadata differ."
            )

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, "wb") as file:
            pickle.dump(self.chunks, file)

    def load(
        self,
        index_path: str,
        metadata_path: str,
    ):
        loaded_index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as file:
            loaded_chunks = pickle.load(file)

        if loaded_index.d != self.dimension:
            raise ValueError(
                f"Index dimension {loaded_index.d} does not match "
                f"configured dimension {self.dimension}."
            )

        if loaded_index.ntotal != len(loaded_chunks):
            raise ValueError(
                "Loaded FAISS vector count does not match "
                "the number of stored chunks."
            )

        self.index = loaded_index
        self.chunks = loaded_chunks
