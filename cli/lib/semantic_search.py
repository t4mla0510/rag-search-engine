from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)


def verify_model():
    sematic_search = SemanticSearch()
    print(f"Model loaded: {EMBEDDING_MODEL}")
    print(f"Max sequence length: {sematic_search.model.max_seq_length}")
