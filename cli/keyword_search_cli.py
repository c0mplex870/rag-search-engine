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

    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")

    args = parser.parse_args()
    
    # Load movies data from JSON file
    data_path = Path(__file__).parent.parent / "data" / "movies.json"
    with open(data_path, "r") as f:
        movies_data = json.load(f)
    # Load stopwords from file
    stopwords_path = Path(__file__).parent.parent / "data" / "stopwords.txt"
    with open(stopwords_path, "r") as f:
        stopwords = f.read().splitlines()

    # Build the inverted index
    stemmer = PorterStemmer()
    idx = InvertedIndex(stopwords, stemmer)
    idx.build(movies_data["movies"])
    
    match args.command:
        case "build":
            idx.save()
            # Get the first document ID for the token 'merida'
            merida_docs = idx.get_documents('merida')
            if merida_docs:
                first_doc_id = merida_docs[0]
                print(f"The first document ID for 'merida' is {first_doc_id}")
        case "search":
            print(f"Searching for: {args.query}")
            
            # Normalize query into non-empty whitespace tokens
            query_tokens = normalize_text(args.query, stopwords, stemmer)
            doc_sets = []
            for token in query_tokens:
                docs = set(idx.get_documents(token))
                if docs:
                    doc_sets.append(docs)
            if not doc_sets:
                print("No results found.")
                return
            else:
                # Intersect document sets for all query tokens
                matched_doc_ids = set.intersection(*doc_sets)
                results = [idx.docmap[doc_id]["title"] for doc_id in matched_doc_ids]
            
            for i, title in enumerate(results[:5], 1):
                print(f"{i}. {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()