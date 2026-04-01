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
    
    chunk_parser = subparsers.add_parser("chunk", help="Split text into word chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk (default: 200)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping words between chunks (default: 0)",
    )
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text into semantic chunks by sentences")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk (default: 4)",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping sentences between chunks (default: 0)",
    )
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
        case "chunk":
            if args.chunk_size <= 0:
                raise ValueError("--chunk-size must be a positive integer")
            if args.overlap < 0:
                raise ValueError("--overlap must be a non-negative integer")
            words = args.text.split()
            
            chunks = [" ".join(words[i : i + args.chunk_size]) for i in range(0, len(words), args.chunk_size)]
            if args.overlap > 0:
                chunks = []
                step = args.chunk_size - args.overlap
                for i in range(0, len(words), step):
                    chunk = " ".join(words[i : i + args.chunk_size])
                    chunks.append(chunk)
            print(f"Chunking {len(args.text)} characters")
            for index, chunk in enumerate(chunks, start=1):
                print(f"{index}. {chunk}")
            
        case "semantic_chunk":
            import re
            
            if args.max_chunk_size <= 0:
                raise ValueError("--max-chunk-size must be a positive integer")
            if args.overlap < 0:
                raise ValueError("--overlap must be a non-negative integer")
            if args.overlap >= args.max_chunk_size:
                raise ValueError("--overlap must be less than --max-chunk-size")

            sentences = re.split(r"(?<=[.!?])\s+", args.text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            chunks = []
            step = max(1, args.max_chunk_size - args.overlap)
            
            for i in range(0, len(sentences), step):
                chunk = sentences[i : i + args.max_chunk_size]
                chunks.append(" ".join(chunk))
                
                if i + args.max_chunk_size >= len(sentences):
                    break
            
            print(f"Chunking {len(args.text)} characters into {len(chunks)} chunks")
            for index, chunk in enumerate(chunks, start=1):
                print(f"{index}. {chunk}")
            
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
