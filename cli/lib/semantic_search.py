import os
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import (
    CACHE_DIR,
    EMBEDDING_MODEL,
    DEFAULT_SEARCH_LIMIT,
    load_movies
)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.embeddings: list[list[float]] = None
        self.documents: list[dict] = None
        self.document_map: dict[int, dict] = {}

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores = []
        for doc_embedding, doc in zip(self.embeddings, self.documents):
            score = cosine_similariy(query_embedding, doc_embedding)
            scores.append((score, doc))
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in sorted_scores[:limit]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })
        return results

    def build_embeddings(self, documents: list[dict]) -> list[list[float]]:
        self.documents = documents
        docs_description = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            text = f"{doc['title']}: {doc['description']}"
            docs_description.append(text)
        self.embeddings = self.model.encode(docs_description, show_progress_bar=True)
        np.save(os.path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> list[list[float]]:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def generate_embedding(self, text: str) -> list[float]:
        if len(text) == 0 or text.isspace():
            raise ValueError("Input text is empty or contains only whitespace.")
        return self.model.encode([text])[0]


def chunk_command(text: str, chunk_size: int) -> None:
    words = text.split()
    chunk = []
    chunks = []
    for word in words:
        chunk.append(word)
        if len(chunk) == chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    print(f"Chunking {chunk_size} characters")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"{idx}. {chunk}\n")


def search_command(query: str, limit: int) -> list[dict]:
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    return semantic_search.search(query, limit)


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text(text: str) -> None:
    semantic_searching = SemanticSearch()
    embeddings = semantic_searching.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embeddings[:3]}")
    print(f"Dimensions: {embeddings.shape[0]}")


def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model() -> None:
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
