import argparse
from lib.hybrid_search import (
    normalize,
    weighted_search_command,
    rrf_search_command,
    fix_spelling,
    rewrite,
    expanding,
    individual_rerank,
    batch_rerank,
    cross_encoder_rerank
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="Available commands")
    
    norm_parser = sub_parser.add_parser("normalize", help="Normalize BM25 and Cosine similarity")
    norm_parser.add_argument("scores", type= float, nargs="+", help="List of floats to normalize")
    
    weighted_search_parser = sub_parser.add_parser("weighted-search", help="Hybrid search with weight")
    weighted_search_parser.add_argument("query", type=str, help="Query to be searched")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, nargs="?", help="Alpha value to balance between keywords and semantic search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, nargs="?", help="Top-k limits")
    
    rrf_search_parser = sub_parser.add_parser("rrf-search", help="RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Query to be searched")
    rrf_search_parser.add_argument("--k", type=int, default=60, nargs="?", help="The k parameter to control weight betweene higher-rank and lower-rank one")
    rrf_search_parser.add_argument("--limit", type=int, default=5, nargs="?", help="Top-k limits")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], default="rewrite", nargs="?", help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], nargs="?", help="Method to rerank")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for idx, res in enumerate(results, start=1):
                print(f"{idx} {res["title"]}")
                print(f"   Hybrid score: {res["hybrid_score"]:.4f}")
                print(f"   BM25: {res["keyword_score"]:.4f}, Semantic: {res["semantic_score"]:.4f}")
                print(f"   {res["description"]}...")
        case "rrf-search":
            search_query = args.query
            if args.enhance:
                enhance_functions = {
                    "spell": fix_spelling,
                    "rewrite": rewrite,
                    "expand": expanding
                }
                enhance_func = enhance_functions.get(args.enhance)
                if enhance_func:
                    search_query = enhance_func(args.query)
                    # Log the original query and the query after enhancements
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{search_query}'\n")
            fetch_limit = args.limit*5 if args.rerank_method else args.limit
            results = rrf_search_command(search_query, args.k, fetch_limit)
            
            # Log the results after rrf search
            for idx, res in enumerate(results, start=1):
                print(f"{idx}. {res["title"]}")
                print(f"   RRF score: {res["rrf_score"]:.4f}")
                print(f"   BM25 Rank: {res["bm25_rank"]}, Semantic Rank: {res["semantic_rank"]}")
                print(f"   {res["description"]}...")
            
            if args.rerank_method:
                rerank_methods = {
                    "individual": individual_rerank,
                    "batch": batch_rerank,
                    "cross_encoder": cross_encoder_rerank
                }
                method = rerank_methods.get(args.rerank_method)
                print(f"Reranking top {args.limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{args.query}' (k={args.k})\n")
                results = method(search_query, results, args.limit)
                # Log the final results after re-ranking
                for idx, res in enumerate(results, start=1):
                    print(f"{idx}. {res["title"]}")
                    if res.get("cross_encoder_score") is not None:
                        print(f"   Cross Encoder Score: {res["cross_encoder_score"]:.4f}")
                    if res.get("rerank_rank") is not None:
                        print(f"   Rerank Rank: {res["rerank_rank"]}")
                    if res.get("rerank_score") is not None:
                        print(f"   Rerank Score: {res["rerank_score"]}/10")
                    print(f"   RRF score: {res["rrf_score"]:.4f}")
                    print(f"   BM25 Rank: {res["bm25_rank"]}, Semantic Rank: {res["semantic_rank"]}")
                    print(f"   {res["description"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
