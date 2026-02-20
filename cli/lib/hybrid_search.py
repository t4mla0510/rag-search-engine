import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents: list[dict] = documents
        self.semantic_search: ChunkedSemanticSearch = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
    
    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> None:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")
    
    def rrf_search(self, query: str, k: float, limit: int = 10) -> None:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
