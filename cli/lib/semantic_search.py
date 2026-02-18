import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    def generate_embedding(self, text: str) -> list[float]:
        if len(text) == 0 or text.isspace():
            raise ValueError("Input text is empty or contains only whitespace.")
        return self.model.encode([text])[0]


def embed_text(text: str) -> list[float]:
    semantic_searching = SemanticSearch()
    embeddings = semantic_searching.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embeddings[:3]}")
    print(f"Dimensions: {embeddings.shape[0]}")


def verify_model():
    sematic_search = SemanticSearch()
    print(f"Model loaded: {EMBEDDING_MODEL}")
    print(f"Max sequence length: {sematic_search.model.max_seq_length}")


def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Dimension mismatch")
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]


def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Dimension mismatch")
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]


def dot(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Dimension mismatch")
    return sum([v1 * v2 for v1, v2 in zip(vec1, vec2)])


def cosine_similariy(vec1: list[float], vec2: list[float]) -> list[float]:
    if len(vec1) != len(vec2):
        raise ValueError("Dimension mismatch")
    dot_product = dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    return dot_product / (magnitude_vec1 * magnitude_vec2)
