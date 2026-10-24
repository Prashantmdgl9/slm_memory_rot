from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def embed(texts, is_query=False):
    model = get_embedder()
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
