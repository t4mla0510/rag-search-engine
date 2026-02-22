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
        
        relevant_retrievd_precision = sum(1 for title in resultt_titles if title in golden_titles)
        relevant_retrievd_recall = sum(1 for title in golden_titles if title in resultt_titles)
        
        precision = relevant_retrievd_precision / len(resultt_titles)
        recall = relevant_retrievd_recall / len(golden_titles)
        f1_score = (2 * precision * recall) / (precision + recall)
        
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Retrieved: {",".join(resultt_titles)}")
        print(f"  - Relevant: {",".join(golden_titles)}")
        

if __name__ == "__main__":
    main()
