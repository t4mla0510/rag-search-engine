import argparse
import json
import string
from typing import List, Dict

def preprocessing(text: str) -> List[str]:
    """Lowercase, remove punctuation, tokenizing"""
    trans_table = str.maketrans('','', string.punctuation)
    cleaned = text.lower().translate(trans_table)
    return cleaned.split()

def load_movies(file_path: str) -> List[Dict]:
    """Load movies from JSON file"""
    with open("./data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("movies", [])

def search_movies(query: str, movies: List[Dict], limit: int = 5) -> List[Dict]:
    """Return movies where at least one query token matches part of any title token."""
    query_tokens = preprocessing(query)
    results = []

    for movie in movies:
        title_tokens = preprocessing(movie.get("title", ""))

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
        print(f"{idx}. {movie.get("title", "")}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            # Load movies
            movies = load_movies("./data/movies.json")
            
            # Find movies that contain query's keywords
            results = search_movies(args.query, movies)

            # Print the result
            print_result(results)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
