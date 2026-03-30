import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
from torch import embedding

class SemanticSearch:
    _model = None

    def __init__(self) -> None:
        if SemanticSearch._model is None:
            cache_folder = Path(__file__).resolve().parents[2] / ".cache" / "sentence_transformers"
            SemanticSearch._model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=str(cache_folder),
            )
        self.model = SemanticSearch._model
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("text must not be empty or whitespace only")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_texts, show_progress_bar=True)

        cache_path = Path(__file__).resolve().parents[2] / "cache" / "movie_embeddings.npy"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        cache_path = Path(__file__).resolve().parents[2] / "cache" / "movie_embeddings.npy"
        if cache_path.exists():
            self.embeddings = np.load(cache_path)
        elif self.embeddings is not None and len(self.embeddings) == len(documents):
            return self.embeddings
        else:
            return self.build_embeddings(documents)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None or self.documents is None:
            raise ValueError("Embeddings or documents not loaded")

        if len(self.embeddings) != len(self.documents):
            raise ValueError("Mismatch")

        query_embedding = self.generate_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        similarities = self.embeddings @ query_embedding

        k = min(limit, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [
            {
                "score": float(similarities[i]),
                "title": self.documents[i].get("title"),
                "description": self.documents[i].get("description"),
            }
            for i in top_indices
            if self.documents[i] is not None
        ]
    

def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    data_path = Path(__file__).resolve().parents[2] / "data" / "movies.json"

    with open(data_path, "r") as file:
        movies_data = json.load(file)

    documents = movies_data["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):  
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)