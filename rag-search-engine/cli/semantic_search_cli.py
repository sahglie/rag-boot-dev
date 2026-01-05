#!/usr/bin/env python3

import argparse
import json
from typing import Any

from semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
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

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=0)

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

        case "chunk":
            chunks = []
            words: list[str] = args.text.split()

            print(f"Chunking {len(args.text)} characters")

            overlap = None
            while len(words):
                if len(words) >= args.chunk_size:
                    chunk = words[: args.chunk_size]
                else:
                    chunk = words

                if args.overlap and overlap:
                    chunk = overlap + chunk

                if args.overlap:
                    overlap = chunk[-args.overlap :]

                chunks.append(" ".join(chunk))
                words = words[args.chunk_size :]

            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

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
