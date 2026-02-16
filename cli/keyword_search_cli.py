import argparse
import json


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
            with open("./data/movies.json", "r", encoding="utf-8") as file:
                data = json.load(file)
            movies = data.get("movies", [])
            
            # Find movies that contain query's keywords
            results = []
            query = args.query
            for movie in movies:
                title = movie.get("title", "")
                if query.lower() in title.lower():
                    results.append(movie)
                    if len(results) == 5:
                        break

            # Print the result
            for idx, movie in enumerate(movies, start=1):
                print(f"{idx}. {movie["title"]}")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
