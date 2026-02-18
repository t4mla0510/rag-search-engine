import argparse

from lib.semantic_search import (
    verify_model,
    embed_text
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    sub_parser = parser.add_subparsers(dest="command", help="All available commands")

    sub_parser.add_parser("verify", help="Verify the embedding model")
    
    embedtext_parser = sub_parser.add_parser("embed_text", help="Get embeddings of input text")
    embedtext_parser.add_argument("text", type=str, help="Text to be embedded")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()