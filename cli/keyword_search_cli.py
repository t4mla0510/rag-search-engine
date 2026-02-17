import os
import json
import pickle
import string
import argparse
from typing import List, Dict

from nltk.stem import PorterStemmer

def load_stopwords(file_path: str) -> List[str]:
    """Load stop words from .txt file"""
    with open(file_path, "r", encoding="utf-8") as file:
        stop_words = file.read().splitlines()
    return stop_words

def preprocessing(text: str) -> List[str]:
    """Lowercase, remove punctuation, tokenizing"""
    trans_table = str.maketrans('','', string.punctuation)
    cleaned = text.lower().translate(trans_table)
    return cleaned.split()

def load_movies(file_path: str) -> List[Dict]:
    """Load movies from JSON file"""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("movies", [])

def search_movies(query: str, movies: List[Dict], stop_words: List[str], stemmer: PorterStemmer, limit: int = 5) -> List[Dict]:
    """Return movies where at least one query token matches part of any title token."""
    query_tokens = preprocessing(query)
    query_tokens = [stemmer.stem(token) for token in query_tokens if token not in stop_words]
    results = []

    for movie in movies:
        title_tokens = preprocessing(movie.get("title", ""))
        title_tokens = [stemmer.stem(token) for token in title_tokens if token not in stop_words]

        match_found = any(
            q_token in t_token
            for q_token in query_tokens
            for t_token in title_tokens
        )

        if match_found:
            results.append(movie)
            if len(results) == limit:
                break
    
    return results

def print_result(results: List[Dict]):
    if not results:
        print("No movies found.")
        return

    for idx, movie in enumerate(results, start=1):
        print(f"{idx}. {movie.get('title', '')}")


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, set[int]] = {}
        self.docmap: Dict[int, Dict] = {}
    
    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = text.lower().split()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    
    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        if term in self.index:
            return sorted(self.index[term])
        return []
    
    def build(self, movies: List[Dict]) -> None:
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie

            text = f"{movie["title"]} {movie["description"]}"
            self._add_document(doc_id, text)

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open("./cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        
        with open("./cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            # Load movies, stop words and stemmer
            movies = load_movies("./data/movies.json")
            stop_words = load_stopwords("./data/stopwords.txt")
            stemmer = PorterStemmer()

            # Find movies that contain query's keywords
            results = search_movies(args.query, movies, stop_words, stemmer)

            # Print the result
            print_result(results)
            pass
        case "build":
            index = InvertedIndex()
            movies = load_movies("./data/movies.json")
            index.build(movies)
            index.save()

            docs = index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
