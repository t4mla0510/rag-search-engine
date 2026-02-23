import os
import argparse
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

from lib.search_utils import LLM_MODEL

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Searching with Image")
    parser.add_argument("--image", required=True, type=str, help="The path to image file")
    parser.add_argument("--query", required=True, type=str, help="A text query to rewrite based on the image")
    
    args = parser.parse_args()
    
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    
    with open(args.image, "rb") as f:
        image_content = f.read()
        
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    system_prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
    parts = [
        system_prompt,
        types.Part.from_bytes(data=image_content, mime_type=mime),
        args.query.strip()
    ]
    
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=parts
    )
    
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
