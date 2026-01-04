#!/usr/bin/env python3

from nltk.stem import PorterStemmer
import string
import argparse
import math
import pprint
from typing import Any
from inverted_index import InvertedIndex


def tokenize(search: str) -> list[str]:
    stemmer = PorterStemmer()
    clean_search = search.translate(str.maketrans("", "", string.punctuation))
    return [stemmer.stem(e).lower() for e in clean_search.split() if e]


def find_matches(search: str, index: InvertedIndex) -> list[dict[str, Any]]:
    doc_ids = set()
    tokens = tokenize(search)
    for t in tokens:
        if t not in index._stopwords:
            # pprint.pprint(t)
            for doc_id in index.get_documents(t):
                doc_ids.add(doc_id)

    movies = []
    for doc_id in sorted(doc_ids):
        movies.append(index._docmap[doc_id])

    return movies


def idf_cmd(term: str, index: InvertedIndex) -> float:
    total_doc_count = len(index._docmap)
    term_match_doc_count = len(index.get_documents(term))

    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def tfidf_cmd(doc_id: int, term: str, index: InvertedIndex) -> float:
    tf = index.get_tf(doc_id, term)
    total_doc_count = len(index._docmap)
    term_match_doc_count = len(index.get_documents(term))
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    return tf * idf


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movie InvertedIndex")

    tf_parser = subparsers.add_parser("tf", help="Search term frequencies")
    tf_parser.add_argument("id", type=int)
    tf_parser.add_argument("term", type=str)

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate inverse document frequency"
    )
    idf_parser.add_argument("term", type=str)

    tfidf_parser = subparsers.add_parser("tfidf", help="Search term frequencies")
    tfidf_parser.add_argument("doc_id", type=int)
    tfidf_parser.add_argument("term", type=str)

    bm25idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=1.5, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=0.75, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # pprint.pprint("hey")

    index = InvertedIndex()
    index.load()

    match args.command:
        case "search":
            matches = find_matches(args.query, index)
            for idx, match in enumerate(matches, 1):
                print(f"{idx}. {match['title']} {match['id']}")
                if idx >= 5:
                    break
        case "bm25search":
            matches = index.bm25_search(args.query, limit=5)
            for idx, match in enumerate(matches, 1):
                print(f"{idx}. ({match['id']}) {match['title']} - {match['score']:.2f}")
        case "build":
            index.build()
            index.save()
        case "tf":
            freq = index.get_tf(args.id, args.term)
            print(freq)
        case "idf":
            idf = idf_cmd(args.term, index)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tfidf = tfidf_cmd(args.doc_id, args.term, index)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}"
            )
        case "bm25idf":
            score = index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {score:.2f}")
        case "bm25tf":
            score = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}': {score:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
