import os
import argparse

from dotenv import load_dotenv
from google import genai

from lib.hybrid_search import rrf_search_command

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query, k=60, limit=5)
            print("Search Results:")
            for res in results:
                print(f"  - {res["title"]}")
            
            documents = [f"{res['id']}. {res['title']}: {res['description']}" for res in results]
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {documents}

            Provide a comprehensive answer that addresses the query:"""
            load_dotenv()
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            print("\nRAG Response:")
            print(response.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
