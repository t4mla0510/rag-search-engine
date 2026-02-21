import argparse
from lib.hybrid_search import (
    normalize,
    weighted_search_command,
    rrf_search_command,
    fix_spelling
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
    
    rrf_search_parser = sub_parser.add_parser("rrf_search", help="RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Query to be searched")
    rrf_search_parser.add_argument("--k", type=int, default=60, nargs="?", help="The k parameter to control weight betweene higher-rank and lower-rank one")
    rrf_search_parser.add_argument("--limit", type=int, default=5, nargs="?", help="Top-k limits")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell"], nargs="?", help="Query enhancement method")

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
                print(f"   BM25: {res["keyword_score"]:.4f}, Semantic: {res["semantic_score"]:.4f}")
                print(f"   {res["description"]}...")
        case "rrf_search":
            if not args.enhance:
                results = rrf_search_command(args.query, args.k, args.limit)
                for idx, res in enumerate(results, start=1):
                    print(f"{idx} {res["title"]}")
                    print(f"   BM25 Rank: {res["bm25_rank"]}, Semantic Rank: {res["semantic_rank"]}")
                    print(f"   RRF score: {res["rrf_score"]:.4f}")
                    print(f"   {res["description"]}...")
            else:
                enhanced_query = fix_spelling(args.query)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{enhanced_query}'\n")
                results = rrf_search_command(enhanced_query, args.k, args.limit)
                for idx, res in enumerate(results, start=1):
                    print(f"{idx} {res["title"]}")
                    print(f"   BM25 Rank: {res["bm25_rank"]}, Semantic Rank: {res["semantic_rank"]}")
                    print(f"   RRF score: {res["rrf_score"]:.4f}")
                    print(f"   {res["description"]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
