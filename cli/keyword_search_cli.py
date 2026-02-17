import argparse

from lib.keyword_search import build_command, search_command, tf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser = subparsers.add_parser("tf", help="Retrieve term frequencies of doc_id and token")
    tf_parser.add_argument("id", type=int, help="The id of document to be retrieved")
    tf_parser.add_argument("term", type=str, help="The term to be retrieved with term frequencies")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            results = tf_command(args.id, args.term)
            print(results) if results else print(0)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
