import os
import math
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
    
    def build(self) -> None:
        movies = load_movies()
        for m in movies:
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

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
    
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
    
    def get_tf(self, doc_id: int, term: str) -> int:
        term_tokens = tokenize_text(term)
        if len(term_tokens) != 1:
            raise ValueError("Term must tokenize to exactly one token.")
        token = term_tokens[0]
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][token]


def build_command():
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
