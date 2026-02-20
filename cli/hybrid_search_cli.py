import argparse
from lib.hybrid_search import (
    normalize,
    weighted_search_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="Available commands")
    
    norm_parser = sub_parser.add_parser("normalize", help="Normalize BM25 and Cosine similarity")
    norm_parser.add_argument("scores", type= float, nargs="+", help="List of floats to normalize")
    
    weighted_search_parser = sub_parser.add_parser("weighted_search", help="Hybrid search with weight")
    weighted_search_parser.add_argument("query", type=str, help="Query to be searched")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, nargs="?", help="Alpha value to balance between keywords and semantic search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, nargs="?", help="Top-k limits")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted_search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for idx, res in enumerate(results, start=1):
                print(f"{idx} {res["title"]}")
                print(f"   Hybrid score: {res["hybrid_score"]:.4f}")
                print(f"   BM25L {res["keyword_score"]:.4f}, Semantic: {res["semantic_score"]:.4f}")
                print(f"   {res["description"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
