import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from .keyword_search import bm25_search_command, schema_keyword_search
from .semantic_search import search_chunks_command, schema_semantic_search
from .hybrid_search import (
    rewrite,
    fix_spelling,
    rrf_search_command,
    schema_rewrite,
    schema_fix_spelling,
    schema_rrf_search
)

load_dotenv()

class MovieAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = "gemini-2.5-flash"
        
        self.tools = [
            types.Tool(
                function_declarations=[
                    schema_rewrite,
                    schema_fix_spelling,
                    schema_keyword_search,
                    schema_semantic_search,
                    schema_rrf_search
                ]
            )
        ]
        
        self.tool_map = {
            "rewrite": lambda query: rewrite(query, self.client),
            "fix_spelling": lambda query: fix_spelling(query, self.client),
            "bm25_search_command": bm25_search_command,
            "search_chunks_command": search_chunks_command,
            "rrf_search": rrf_search_command
        }
        
    def run(self, user_query: str):
        print(f"[USER]: {user_query}")
        
        chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                tools=self.tools,
                system_instruction="""
                You are the Hoopla Movie Assistant. Help users find movies using this workflow:
                1. PRE-PROCESS: Use 'fix_spelling' for typos or 'rewrite' for vague queries (e.g., "that bear movie").
                2. SEARCH: 
                - 'rrf_search' for general requests.
                - 'bm25_search_command' for specific titles/names.
                - 'search_chunks_command' for plot details or "vibes."
                3. ANSWER: Use only the provided movie documents to answer the user.

                Style Guide:
                - Be direct, casual, and conversational (like a normal person in chat).
                - Cite titles clearly.
                - No "cringe" or "hype" (avoid: "Experience the thrill of...").
                - If no results, suggest a new 'rewrite' once.
                """
            )
        )
        
        response = chat.send_message(user_query)
        
        while response.candidates[0].content.parts[0].function_call:
            tool_responses = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    call = part.function_call
                    print(f" --> [Agent] Calling Tool: {call.name} with args {call.args}")
                    
                    function_to_call = self.tool_map[call.name]
                    tool_result = function_to_call(**call.args)
                    
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": tool_result}
                        )
                    )
            response = chat.send_message(tool_responses)
                    
        print(f"\n[FINAL RESPONSE]: {response.text}")
