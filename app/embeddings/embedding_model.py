from typing import List 
from sentence_transformers import SentenceTransformer
from app.ingestion.code_chunker import CodeChunk 

class CodeEmbeddingModel:
    def __init__(
            self,
            model_name:str = "BAAI/bge-small-en-v1.5"

    ):
        print(f"Loading embedding model:{model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model Loaded successfully.")


    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single piece of text into an embedding.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)

        return embedding.tolist()

    def embed_chunks(self,chunks: List[CodeChunk]) -> List[List[float]]:
        """
        Convert multiple CodeChunks into embeddings.
        """
        if not chunks:
            return []
        texts = [self.prepare_chunk_text(chunk) for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()
    def prepare_chunk_text(self, chunk: CodeChunk) -> str:
        """
        Prepare the text representation of a CodeChunk for embedding.
        """
        return (
            f"File: {chunk.file_path}\n"
            f"Symbol: {chunk.symbol_name}\n"
            f"Type: {chunk.symbol_type}\n"
            f"Code:\n{chunk.content}"
        )
        



