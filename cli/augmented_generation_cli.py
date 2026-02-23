import os
import argparse

from dotenv import load_dotenv
from google import genai

from lib.hybrid_search import rrf_search_command
from lib.search_utils import LLM_MODEL

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser("summarize", help="Perform RAG summary")
    summarize_parser.add_argument("query", type=str, help="Search query for RAG summary")
    summarize_parser.add_argument("--limit", type=int, default=5, nargs="?", help="Search query for RAG summary")
    
    citation_parser = subparsers.add_parser("citations", help="Perform RAG summary with citations")
    citation_parser.add_argument("query", type=str, help="Search query for RAG summary with citations")
    citation_parser.add_argument("--limit", type=int, default=5, nargs="?", help="Search limits")
    
    question_parser = subparsers.add_parser("question", help="Perform Q&A with RAG")
    question_parser.add_argument("question", type=str, help="Question for Q&A RAG")
    question_parser.add_argument("--limit", type=int, default=5, nargs="?", help="Search limits")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query)
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
                model=LLM_MODEL,
                contents=prompt
            )
            print("\nRAG Response:")
            print(response.text)
        case "summarize":
            query = args.query
            results = rrf_search_command(query)
            print("Search Results:")
            for res in results:
                print(f"  - {res["title"]}")
            
            prompt = f"""
            Provide information useful to this query by synthesizing information from multiple search results in detail.
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Search Results:
            {results}
            Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
            """
            load_dotenv()
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            print("\nLLM Summary:")
            print(response.text)
        case "citations":
            query = args.query
            results = rrf_search_command(query)
            print("Search Results:")
            for res in results:
                print(f"  {res["id"]}. {res["title"]}")
            
            documents = [f"{res['id']}. {res['title']}: {res['description']}" for res in results]
            prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {documents}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
            load_dotenv()
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            print("\nLLM Answer:")
            print(response.text)
        case "question":
            question = args.question
            results = rrf_search_command(question)
            print("Search Results:")
            for res in results:
                print(f"  - {res["title"]}")
            
            context = "\n".join([f"{res['id']}. {res['title']}: {res['description']}" for res in results])
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Question: {question}

            Documents:
            {context}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
            load_dotenv()
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            print("\nAnswer:")
            print(response.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
