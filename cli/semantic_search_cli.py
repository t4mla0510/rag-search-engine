import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_command,
    sentence_chunk_command,
    embed_chunks_command,
    search_chunks_command
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="All available commands")

    sub_parser.add_parser("verify", help="Verify the embedding model")
    
    embedtext_parser = sub_parser.add_parser("embed-text", help="Get embeddings of input text")
    embedtext_parser.add_argument("text", type=str, help="Text to be embedded")

    sub_parser.add_parser("verify-embeddings", help="Verify generated embeddings")

    embedquery_parser = sub_parser.add_parser("embed-query", help="Get embeddings of query")
    embedquery_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = sub_parser.add_parser("search", help="Search with cosine similarity")
    search_parser.add_argument("query", type=str, help="Query to be searched")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Top-k limits")

    chunking_parser = sub_parser.add_parser("chunk", help="Chunking given text")
    chunking_parser.add_argument("text", type=str, help="Text to be chunked")
    chunking_parser.add_argument("--chunk_size", type=int, nargs="?", default=100, help="Size of chunk")
    chunking_parser.add_argument("--overlap", type=int, nargs="?", default=20, help="Overlap between chunks")

    sentence_parser = sub_parser.add_parser("sentence-chunk", help="Semantic chunking given text")
    sentence_parser.add_argument("text", type=str, help="Text to be chunked")
    sentence_parser.add_argument("--max_chunk_size", type=int, nargs="?", default=4, help="Size of chunk")
    sentence_parser.add_argument("--overlap", type=int, nargs="?", default=0, help="Overlap between chunks")

    sub_parser.add_parser("embed-chunks", help="Create chunk embeddings of all documents")
    
    search_chunked_parser = sub_parser.add_parser("search-chunks", help="Search relevant chunks")
    search_chunked_parser.add_argument("query", type=str, help="Query to be searched")
    search_chunked_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Top-k chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed-text":
            embed_text(args.text)
        case "verify-embeddings":
            verify_embeddings()
        case "embed-query":
            embed_query_text(args.query)
        case "search":
            movies = search_command(args.query, args.limit)
            for idx, movie in enumerate(movies, start=1):
                print(f"{idx}. {movie["title"]} (score: {movie["score"]:.4f})")
                print(f"   {movie["description"][:100]} ...")
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "sentence-chunk":
            chunks = sentence_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(f"Sentence chunking {len(chunks)} chunks")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")
        case "embed-chunks":
            embed_chunks_command()
        case "search-chunks":
            relevant_chunks = search_chunks_command(args.query, args.limit)
            for idx, chunk in enumerate(relevant_chunks, start=1):
                print(f"\n{idx}. {chunk["title"]} (score: {chunk["score"]:.4f})")
                print(f"   {chunk["description"]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()