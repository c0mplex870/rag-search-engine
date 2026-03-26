#!/usr/bin/env python3

import argparse
import json
import string
from pathlib import Path
from nltk.stem import PorterStemmer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from inverted_index import InvertedIndex

def normalize_text(text: str, stopwords: list[str], stemmer: PorterStemmer) -> list[str]:
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    tokens = [token for token in clean_text.split() if token and token not in stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    
    args = parser.parse_args()
    
    # Load stopwords from file
    stopwords_path = Path(__file__).parent.parent / "data" / "stopwords.txt"
    with open(stopwords_path, "r") as f:
        stopwords = f.read().splitlines()

    stemmer = PorterStemmer()
    
    match args.command:
        case "build":
            # Load movies data from JSON file
            data_path = Path(__file__).parent.parent / "data" / "movies.json"
            with open(data_path, "r") as f:
                movies_data = json.load(f)
            
            # Build the inverted index
            idx = InvertedIndex(stopwords, stemmer)
            idx.build(movies_data["movies"])
            idx.save()
            
        case "search":
            # Load the inverted index from disk
            idx = InvertedIndex(stopwords, stemmer)
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            
            print(f"Searching for: {args.query}")
            
            # Normalize query into tokens
            query_tokens = normalize_text(args.query, stopwords, stemmer)
            
            # Collect matching document IDs (stop after 5)
            matched_doc_ids = set()
            for token in query_tokens:
                if len(matched_doc_ids) >= 5:
                    break
                docs = idx.get_documents(token)
                matched_doc_ids.update(docs)
            
            # Print results (title and ID, max 5)
            for i, doc_id in enumerate(sorted(matched_doc_ids)[:5], 1):
                title = idx.docmap[doc_id]["title"]
                print(f"{i}. {title} (ID: {doc_id})")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()