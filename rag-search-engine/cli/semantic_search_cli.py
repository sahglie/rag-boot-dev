#!/usr/bin/env python3

import argparse
import json
from typing import Any
from semantic_search import (
    SemanticSearch,
    verify_embeddings,
    verify_model,
    embed_text,
    embed_query_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify semantic search model")
    subparsers.add_parser("verify_embeddings", help="Verify semantic search embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Embed text")
    embed_parser.add_argument("text", type=str)

    embed_query_parser = subparsers.add_parser("embedquery", help="Embed query")
    embed_query_parser.add_argument("text", type=str)

    search_parser = subparsers.add_parser("search", help="search")
    search_parser.add_argument("text", type=str)
    search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "search":
            search = SemanticSearch()

            with open("data/movies.json") as f:
                json_data: dict[str, list[dict[str, Any]]] = json.load(f)
                movies = json_data["movies"]
                search.load_or_create_embeddings(movies)

            movies = search.search(args.text, 5)
            for idx, m in enumerate(movies, 1):
                print(
                    f"{idx}. {m['title']} (score: {m['score']:.4f})\n{m['description'][:100]}...\n"
                )

        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
