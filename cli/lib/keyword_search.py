import os
import math
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer
from tqdm import tqdm
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
    
    def build(self) -> None:
        movies = load_movies()
        for m in tqdm(movies):
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self._add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
    
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise ValueError("Term must tokenize to exactly one token.")
        token = term_tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][token]

    def get_bm25_idf(self, term: str) -> float:
        term_tokens = tokenize_text(term)
        valid_tokens = []
        for token in term_tokens:
            if not token:
                raise ValueError("Term must tokenize to exactly one token.")
            valid_tokens.append(token)
        token = valid_tokens[0]
        nums_docs = len(self.docmap)
        docs_freq = len(self.index[token])
        return math.log((nums_docs - docs_freq + 0.5) / (docs_freq + 0.5) + 1)
    
    def _get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self._get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query: str, limit: int) -> list[dict]:
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            result = {
                "doc_id": doc['id'],
                "title": doc['title'],
                "document": doc['description'],
                "score": score
            }
            results.append(result)
        return results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    seen, results = set(), []
    for query_token in query_tokens:
        matching_doc_ids = idx.get_documents(query_token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = idx.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    return tf


def idf_command(term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    term_tokens = tokenize_text(term)
    if len(term_tokens) != 1:
        raise ValueError("IDF can only calculated for a single token.")
    token = term_tokens[0]
    nums_docs = len(idx.docmap)
    docs_freq = len(idx.index.get(token, []))
    idf = math.log((nums_docs + 1) / (docs_freq + 1))
    return idf


def tfidf_command(doc_id: int, term: str) -> float:
    tf = tf_command(doc_id, term)
    idf = idf_command(term)
    tfidf = tf * idf
    return tfidf


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    bm25_idf = idx.get_bm25_idf(term)
    return bm25_idf


def bm25_tf_command(doc_id: int, term: str, k1: float) -> float:
    idx = InvertedIndex()
    idx.load()
    if k1 != BM25_K1:
        bm25_tf = idx.get_bm25_tf(doc_id, term, k1)
    else:
        bm25_tf = idx.get_bm25_tf(doc_id, term)
    return bm25_tf


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = [word for word in valid_tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return stemmed_words
