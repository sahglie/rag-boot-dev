from nltk.stem import PorterStemmer
from typing import Any
import json
import string
import pickle
from pathlib import Path
from collections import Counter
import math
import pprint


class InvertedIndex:
    def __init__(self) -> None:
        self._index: dict[str, set[int]] = {}
        self._docmap: dict[int, Any] = {}
        self._doc_lengths: dict[int, int] = {}
        self._term_frequencies: dict[int, Counter] = {}
        self._stemmer = PorterStemmer()
        self._stopwords: set[str] = set()
        self._movies: list[dict[str, Any]] = []

    def get_documents(self, term: str) -> list[int]:
        tokens = self.__tokenize(term)
        doc_ids = self._index.get(tokens[0], set())
        return sorted(doc_ids)

    def bm25(self, doc_id, term) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int) -> list[Any]:
        tokens = self.__tokenize(query)
        scores: dict[int, float] = {}
        for token in tokens:
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                score = self.bm25(doc_id, token)
                if scores.get(doc_id):
                    scores[doc_id] += score
                else:
                    scores[doc_id] = score

        docs = []
        scored_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for doc_id, score in scored_docs:
            d = self._docmap[doc_id]
            d["score"] = score
            docs.append(self._docmap[doc_id])
            if len(docs) >= limit:
                break

        return docs

    def get_tf(self, doc_id, term) -> int:
        tokens = self.__tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must have a single term")

        counter = self._term_frequencies[doc_id]
        return counter[tokens[0]]

    def get_bm25_tf(self, doc_id: int, term: str, k1=1.5, b=0.75) -> float:
        tokens = self.__tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must have a single term")

        doc_length = self._doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        len_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, tokens[0])

        return (tf * (k1 + 1)) / (tf + k1 * len_norm)
        # return (tf * (k1 + 1)) / (tf + k1)

    def get_bm25_idf(self, term: str) -> float:
        tokens = self.__tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must have a single term")

        df = len(self.get_documents(tokens[0]))
        N = len(self._docmap)

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def build(self) -> None:
        self.__load_stopwords()
        self.__load_movies()

        for m in self._movies:
            self._docmap[m["id"]] = m
            self.__add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self) -> None:
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self._index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self._docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self._term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self._doc_lengths, f)

    def load(self) -> None:
        with open("cache/index.pkl", "rb") as f:
            self._index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self._docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self._term_frequencies = pickle.load(f)
        with open("cache/doc_lengths.pkl", "rb") as f:
            self._doc_lengths = pickle.load(f)

    def __tokenize(self, text) -> list[str]:
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = []
        for t in text.lower().split():
            if t and t not in self._stopwords:
                tokens.append(self._stemmer.stem(t))
        return tokens

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.__tokenize(text)
        for t in tokens:
            self._index.setdefault(t, set()).add(doc_id)

        self._term_frequencies[doc_id] = Counter(tokens)
        self._doc_lengths[doc_id] = len(tokens)

    def __load_stopwords(self) -> None:
        with open("data/stopwords.txt") as f:
            self._stopwords = {w.strip() for w in f.readlines()}

    def __load_movies(self) -> None:
        with open("data/movies.json") as f:
            json_data: dict[str, list[dict[str, Any]]] = json.load(f)
            self._movies = json_data["movies"]

    def __get_avg_doc_length(self) -> float:
        # assert self._docmap.keys() == self._doc_lengths.keys()

        num_docs = len(self._docmap)
        if num_docs == 0:
            return 0.0

        total: float = 0.0
        for doc_id in self._docmap:
            total += self._doc_lengths[doc_id]

        return total / num_docs
