import os
from dotenv import load_dotenv

from google import genai

from .search_utils import load_movies, LLM_MODEL
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
        bm25_raw = self._bm25_search(query, limit=limit*500)
        semantic_raw = self.semantic_search.search_chunks(query, limit=limit*500)
        
        if bm25_raw:
            bm25_scores = [r["score"] for r in bm25_raw]
            norm_bm25_values = normalize(bm25_scores)
            for i, res in enumerate(bm25_raw):
                res["norm_score"] = norm_bm25_values[i]
        
        if semantic_raw:
            semantic_scores = [r["score"] for r in semantic_raw]
            norm_semantic_values = normalize(semantic_scores)
            for i, res in enumerate(semantic_raw):
                res["norm_score"] = norm_semantic_values[i]
        
        score_mapping = {}
        for res in bm25_raw:
            uid = res["doc_id"]
            score_mapping[uid] = {
                "title": res["title"],
                "description": res["document"],
                "k_score": res["norm_score"],
                "s_score": 0.0
            }
        for res in semantic_raw:
            uid = res["id"]
            if uid in score_mapping:
                score_mapping[uid]["s_score"] = res["norm_score"]
            else:
                score_mapping[uid] = {
                    "title": res["title"],
                    "description": res["description"],
                    "k_score": 0.0,
                    "s_score": res["norm_score"]
                }

        hybrid_results = []
        for uid, data in score_mapping.items():
            final_score = hybrid_score(data["k_score"], data["s_score"], alpha)
            hybrid_results.append({
                "id": uid,
                "title": data["title"],
                "description": data["description"][:100],
                "hybrid_score": final_score,
                "keyword_score": data["k_score"],
                "semantic_score": data["s_score"]
            })
            
        sorted_results = sorted(hybrid_results, key=lambda x: x["hybrid_score"], reverse=True)
        return sorted_results[:limit]
        
    def rrf_search(self, query: str, k: float, limit: int = 10) -> None:
        bm25_raw = self._bm25_search(query, limit=limit*500)
        semantic_raw = self.semantic_search.search_chunks(query, limit=limit*500)
        
        score_mapping = {}
        for i, res in enumerate(bm25_raw):
            doc_id = res["doc_id"]
            rank = i + 1
            score = rrf_score(rank)
            score_mapping[doc_id] = {
                "title": res["title"],
                "description": res["document"],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_total": score
            }
        for i, res in enumerate(semantic_raw):
            doc_id = res["id"]
            rank = i + 1
            score = rrf_score(rank)
            if doc_id in score_mapping:
                score_mapping[doc_id]["rrf_total"] += score
                score_mapping[doc_id]["semantic_rank"] = rank
            else:
                score_mapping[doc_id] = {
                    "title": res["title"],
                    "description": res["description"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_total": score
                }
        
        final_results = []
        for uid, data in score_mapping.items():
            final_results.append({
                "id": uid,
                "title": data["title"],
                "description": data["description"][:100],
                "bm25_rank": data["bm25_rank"],
                "semantic_rank": data["semantic_rank"],
                "rrf_score": data["rrf_total"]
            })
        sorted_results = sorted(final_results, key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results[:limit]


def expanding(query: str) -> str:
    system_prompt = f"""Expand this movie search query with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.

    Examples:

    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=system_prompt
    )
    return response.text


def rewrite(query: str) -> str:
    system_prompt = f"""Rewrite this movie search query to be more specific and searchable.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:

    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=system_prompt
    )
    return response.text


def fix_spelling(query: str) -> str:
    system_prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=system_prompt
    )
    return response.text


def rrf_search_command(query: str, k: float, limit: int) -> list[dict]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search.rrf_search(query, k, limit)
    

def weighted_search_command(query: str, alpha: float, limit: float) -> list[dict]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search.weighted_search(query, alpha, limit)
    

def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    min_value = min(values)
    if max_value == min_value:
        return [1.0] * len(values)
    return [(value - min_value) / (max_value - min_value) for value in values]


def hybrid_score(bm25_score, semantic_score, alpha=0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)