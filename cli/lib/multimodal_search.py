import os
import torch
import pickle
from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_movies, CACHE_DIR

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict] = None
        self.text_embeddings: list[torch.Tensor] = None

    def load_or_create_embeddings(self, documents: list[dict]) -> list[list[float]]:
        self.documents = documents
        documents_embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings_tensor.pt")
        
        if os.path.exists(documents_embeddings_path):
            self.text_embeddings = torch.load(documents_embeddings_path)
            return self.text_embeddings
        
        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        torch.save(embeddings, documents_embeddings_path)
        self.text_embeddings = embeddings
        return self.text_embeddings
    
    def search_with_image(self, image_path: str) -> list[dict]:
        image = Image.open(image_path)
        image_embedding = self.model.encode([image], convert_to_tensor=True)

        results = []
        for idx, embedding in enumerate(self.text_embeddings):
            doc = self.documents[idx]
            score = torch.cosine_similarity(image_embedding, embedding)
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"][:100],
                "score": float(score)
            })
        sorted_results = sorted(results, key= lambda x: x["score"], reverse=True)
        return sorted_results[:5]

    def embed_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        return self.model.encode([image])[0]


def image_search_command(image_path: str) -> list[dict]:
    documents = load_movies()
    multimodal_search = MultimodalSearch()
    multimodal_search.load_or_create_embeddings(documents)
    return multimodal_search.search_with_image(image_path)


def verify_image_embedding(image_path: str) -> None:
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
