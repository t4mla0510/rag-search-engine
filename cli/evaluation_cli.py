import argparse
import json

from lib.hybrid_search import rrf_search_command

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        evaluation_dataset = json.load(f)
    
    for case in evaluation_dataset["test_cases"]:
        query = case["query"]
        golden_titles = case["relevant_docs"]
        results = rrf_search_command(query, k=60, limit=limit)
        resultt_titles = [res["title"] for res in results]
        
        relevant_found = sum(1 for title in resultt_titles if title in golden_titles)
        precision = relevant_found / limit if limit > 0 else 0.0
        print(f"- Query: {query}")
        print(f"   - Precision@{limit}: {precision}")
        print( f"  - Retrieved: {",".join(resultt_titles)}")
        print( f"  - Relevant: {",".join(golden_titles)}")
        

if __name__ == "__main__":
    main()
