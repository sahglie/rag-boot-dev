from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import Any
from pathlib import Path


class SemanticSearch:
    MOVIE_EMBEDDINGS_FILE = "cache/movie_embeddings.npy"

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("text can't be empty")

        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]) -> list:
        self.documents = documents

        docs = []
        for d in documents:
            self.document_map[d["id"]] = d
            docs.append(f"{d['title']}: {d['description']}")

        self.embeddings = self.model.encode(docs, show_progress_bar=True)
        np.save(self.MOVIE_EMBEDDINGS_FILE, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        docs = []
        for d in documents:
            self.document_map[d["id"]] = d
            docs.append(f"{d['title']}: {d['description']}")

        file_path = Path(self.MOVIE_EMBEDDINGS_FILE)
        if file_path.exists():
            self.embeddings = np.load(self.MOVIE_EMBEDDINGS_FILE)
            if len(self.embeddings) != len(self.documents):
                self.build_embeddings(documents)
        else:
            self.build_embeddings(documents)

        return self.embeddings


def verify_model():
    search = SemanticSearch()
    model = search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")


def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()

    with open("data/movies.json") as f:
        json_data: dict[str, list[dict[str, Any]]] = json.load(f)
        movies = json_data["movies"]
        search.load_or_create_embeddings(movies)

    documents = search.documents
    embeddings = search.embeddings

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
