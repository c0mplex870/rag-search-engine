#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the model loads correctly")
    subparsers.add_parser("verify_embeddings", help="Verify that movie embeddings load correctly")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for input text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Query string to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "verify":
            from lib.semantic_search import verify_model

            verify_model()
        case "verify_embeddings":
            from lib.semantic_search import verify_embeddings

            verify_embeddings()
        case "embed_text":
            from lib.semantic_search import embed_text

            embed_text(args.text)
        case "embedquery":
            from lib.semantic_search import embed_query_text

            embed_query_text(args.query)
        case "search":
            from lib.semantic_search import SemanticSearch

            if args.limit <= 0:
                raise ValueError("--limit must be a positive integer")

            data_path = Path(__file__).resolve().parents[1] / "data" / "movies.json"
            with open(data_path, "r") as file:
                movies_data = json.load(file)

            semantic_search = SemanticSearch()
            documents = movies_data["movies"]
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)

            for index, result in enumerate(results, start=1):
                description = result["description"]
                if len(description) > 100:
                    description = description[:97] + "..."

                print(f"{index}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {description}")
                if index != len(results):
                    print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
