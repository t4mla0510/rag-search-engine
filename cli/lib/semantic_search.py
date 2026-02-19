import os
import re
import json
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings: list[list[float]] = None
        self.chunk_metadata: list[dict] = None

    def build_chunk_embeddings(self, documents: list[dict]) -> list[list[float]]:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks = []
        chunk_metadata = []
        for movie_idx, doc in enumerate(documents):
            if len(doc["description"]) == 0 or doc["description"].isspace():
                continue
            doc_chunks = sentence_chunk_command(doc["description"], 4, 1)
            total_chunks = len(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(os.path.join(CACHE_DIR, "chunk_embeddings.npy"), self.chunk_embeddings)
        with open(os.path.join(CACHE_DIR, "chunk_metadata.json"), "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> list[list[float]]:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")
        if os.path.exists(chunk_embeddings_path) and os.path.exists(metadata_path):
            self.chunk_embeddings = np.load(chunk_embeddings_path)
            with open(metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)


def embed_chunks_command():
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def sentence_chunk_command(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    step = max_chunk_size - overlap if overlap > 0 else max_chunk_size
    for i in range(0, len(sentences), step):
        chunk = sentences[i:i + max_chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if i + max_chunk_size >= len(sentences):
            break
    return chunks


def chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    words = text.split()
    chunks = []
    step = chunk_size - overlap if overlap > 0 else chunk_size
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
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
