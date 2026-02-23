import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Search")
    sub_parser = parser.add_subparsers(dest="command", help="List of available commands")
    
    image_embedding_parser = sub_parser.add_parser("verify_image_embedding", help="Get embeddings of an image")
    image_embedding_parser.add_argument("image_path", type=str, help="Image path")

    image_search_parser = sub_parser.add_parser("image_search", help="Search movies with image")
    image_search_parser.add_argument("image_path", type=str, help="Image path")
    
    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for idx, res in enumerate(results, start=1):
                print(f"{idx}. {res['title']} (similarity: {res['score']:.4f})")
                print(f"   {res['description']}...")
        case _:
            parser.print_help()