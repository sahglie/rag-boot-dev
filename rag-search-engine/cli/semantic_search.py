import json
import pprint
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    MOVIE_EMBEDDINGS = "cache/movie_embeddings.npy"
    MOVIE_CHUNK_EMBEDDINGS = "cache/chunk_embeddings.npy"
    CHUNK_METADATA_JSON = "cache/chunk_metadata.json"
    MOVIES_JSON = "data/movies.json"
    SENTENCE_REGEXP = r"(?<=[.!?])\s+"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray = []
        self.documents: list[dict[int | str, Any]] = []
        self.document_map: dict[int | str, Any] = {}

    def search(self, query: str, limit=5) -> list[dict[Any, Any]]:
        if not len(self.embeddings):
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        scores = []
        for i, d in enumerate(self.documents):
            score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((score, d))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, d in scores[:limit]:
            results.append(
                {
                    "score": score,
                    "title": d["title"],
                    "description": d["description"],
                }
            )

        return results

    def generate_embedding(self, text: str) -> np.ndarray:
        if not len(text.strip()):
            raise ValueError("text can't be empty")

        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict[Any, Any]]) -> np.ndarray:
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}

        documents_data = [f"{d['title']}: {d['description']}" for d in documents]
        self.embeddings = self.model.encode(documents_data, show_progress_bar=True)

        path = Path(self.MOVIE_EMBEDDINGS)
        np.save(path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[Any, Any]]) -> np.ndarray:
        path = Path(self.MOVIE_EMBEDDINGS)

        if path.exists():
            self.documents = documents
            self.document_map = {d["id"]: d for d in documents}
            self.embeddings = np.load(path)
        else:
            self.build_embeddings(documents)

        return self.embeddings


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray = []
        self.chunk_metadata: list[dict[str, Any]] = []

    def search(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)

        chunk_scores: list[dict[str, Any]] = []
        for ce_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk_embedding, query_embedding)
            meta = self.chunk_metadata[ce_idx]
            chunk_scores.append(
                {
                    "score": score,
                    "chunk_idx": ce_idx,
                    "movie_idx": meta["movie_idx"],
                }
            )

        chunk_scores.sort(key=lambda x: x["score"], reverse=True)

        docs = {}
        for cs in chunk_scores:
            doc = self.documents[cs["movie_idx"]]

            if doc["id"] in docs:
                continue
            else:
                docs[doc["id"]] = {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"][:100],
                    "score": cs["score"],
                    "metadata": self.chunk_metadata[cs["chunk_idx"]],
                }

        movies = list(docs.values())
        movies.sort(key=lambda x: x["score"], reverse=True)

        return movies[:limit]

    def build_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}

        all_chunks: list[str] = []
        chunk_metadata = []

        for idx, d in enumerate(self.documents):
            if d["description"].strip():
                chunks = chunk(
                    d["description"],
                    regexp=self.SENTENCE_REGEXP,
                    chunk_size=4,
                    overlap=1,
                )

                for i, c in enumerate(chunks):
                    all_chunks.append(c)
                    chunk_metadata.append(
                        {
                            "movie_idx": idx,
                            "chunk_idx": i,
                            "total_chunks": len(chunks),
                        }
                    )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(self.MOVIE_CHUNK_EMBEDDINGS, self.chunk_embeddings)

        with open(self.CHUNK_METADATA_JSON, "w") as fd:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)},
                fd,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list):
        self.documents = documents
        self.document_map = {d["id"]: d for d in documents}

        chunk_path = Path(self.MOVIE_CHUNK_EMBEDDINGS)
        metadata_path = Path(self.CHUNK_METADATA_JSON)

        if (not chunk_path.exists()) or (not metadata_path.exists()):
            self.build_chunk_embeddings(documents)
        else:
            self.chunk_embeddings = np.load(chunk_path)
            self.chunk_metadata = load_chunk_metadata()

        return self.chunk_embeddings


def load_movies() -> list[dict[Any, Any]]:
    with open(SemanticSearch.MOVIES_JSON) as fd:
        return json.load(fd)["movies"]


def load_chunk_metadata() -> list[dict[Any, Any]]:
    with open(ChunkedSemanticSearch.CHUNK_METADATA_JSON) as fd:
        return json.load(fd)["chunks"]


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()
    movies = load_movies()
    search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(search.documents)}")
    print(
        f"Embeddings shape: {search.embeddings.shape[0]} vectors in {search.embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(v1, v2) -> float:
    dp = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0.0

    return dp / (n1 * n2)


def chunk(text: str, regexp=None, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    if regexp is None:
        tokens = text.split()
    else:
        tokens = re.split(regexp, text)

    chunks = []

    i = 0
    while True:
        if len(tokens) - i > chunk_size:
            chunks.append(" ".join(tokens[i : chunk_size + i]))
            i += chunk_size - overlap
        else:
            chunks.append(" ".join(tokens[i:]))
            break

    return chunks
