import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="All available commands")

    sub_parser.add_parser("verify", help="Verify the embedding model")
    
    embedtext_parser = sub_parser.add_parser("embed_text", help="Get embeddings of input text")
    embedtext_parser.add_argument("text", type=str, help="Text to be embedded")

    sub_parser.add_parser("verify_embeddings", help="Verify generated embeddings")

    embedquery_parser = sub_parser.add_parser("embed_query", help="Get embeddings of query")
    embedquery_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = sub_parser.add_parser("search", help="Search with cosine similarity")
    search_parser.add_argument("query", type=str, help="Query to be searched")
    search_parser.add_argument("limit", type=int, nargs="?", default=5, help="Top-k limits")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            movies = search_command(args.query, args.limit)
            for idx, movie in enumerate(movies, start=1):
                print(f"{idx}. {movie["title"]} (score: {movie["score"]:.4f})")
                print(f"   {movie["description"][:100]} ...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()