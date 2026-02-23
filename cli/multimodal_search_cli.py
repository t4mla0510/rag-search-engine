import argparse
from lib.multimodal_search import verify_image_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Search")
    sub_parser = parser.add_subparsers(dest="command", help="List of available commands")
    
    image_embedding_parser = sub_parser.add_parser("verify_image_embedding", help="Get embeddings of an image")
    image_embedding_parser.add_argument("image_path", type=str, help="Image path")
    
    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()