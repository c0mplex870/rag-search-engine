#!/usr/bin/env python3

import argparse
import json
import string
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    # Load movies data from JSON file
    data_path = Path(__file__).parent.parent / "data" / "movies.json"
    with open(data_path, "r") as f:
        movies_data = json.load(f)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            
            # Filter movies by title
            results = []
            
            # Create translation table to remove punctuation
            translator = str.maketrans('', '', string.punctuation)
            
            # Remove punctuation and lowercase the query
            query_clean = args.query.translate(translator).lower()
            
            for movie in movies_data["movies"]:
                # Remove punctuation and lowercase the title
                title_clean = movie["title"].translate(translator).lower()
                if query_clean in title_clean:
                    results.append(movie["title"])
            
            # Print results (max 5)
            for i, title in enumerate(results[:5], 1):
                print(f"{i}. {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()