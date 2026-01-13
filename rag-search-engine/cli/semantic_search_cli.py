#!/usr/bin/env python3

import argparse
import pprint

from semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    chunk,
    embed_query_text,
    embed_text,
    load_movies,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify")

    embed_parser = subparsers.add_parser("embed_text")
    embed_parser.add_argument("text")

    subparsers.add_parser("verify_embeddings")

    embed_query_parser = subparsers.add_parser("embedquery")
    embed_query_parser.add_argument("text")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text")
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=0)

    sem_chunk_parser = subparsers.add_parser("semantic_chunk")
    sem_chunk_parser.add_argument("text")
    sem_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    sem_chunk_parser.add_argument("--overlap", type=int, default=0)

    embed_chunk_parser = subparsers.add_parser("embed_chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.text)
        case "search":
            movies = load_movies()
            search = SemanticSearch()
            search.load_or_create_embeddings(movies)

            records = search.search(args.query, args.limit)
            for idx, r in enumerate(records, 1):
                print(f"{idx}. {r['title']} (score: {r['score']:.4f})")
        case "chunk":
            chunks = chunk(args.text, chunk_size=args.chunk_size, overlap=args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {c}")
        case "semantic_chunk":
            chunks = chunk(
                args.text,
                regexp=r"(?<=[.!?])\s+",
                chunk_size=args.max_chunk_size,
                overlap=args.overlap,
            )
            print(f"Semantically chunking {len(args.text)} characters")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {c}")
        case "embed_chunks":
            movies = load_movies()
            search = ChunkedSemanticSearch()
            search.load_or_create_chunk_embeddings(movies)

            print(f"Generated {len(search.chunk_embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
