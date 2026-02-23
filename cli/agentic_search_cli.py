import argparse
from lib.agentic_search import MovieAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic search with RAG")
    parser.add_argument("--query", required=True, type=str, help="Query to be searched")
    
    args = parser.parse_args()
    
    agent = MovieAgent()
    agent.run(args.query)
