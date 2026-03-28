#!/usr/bin/env python3

import argparse
import json
import math
import string
from pathlib import Path
from nltk.stem import PorterStemmer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from constants import BM25_B, BM25_K1
from inverted_index import InvertedIndex

def normalize_text(text: str, stopwords: list[str], stemmer: PorterStemmer) -> list[str]:
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()
    tokens = [token for token in clean_text.split() if token and token not in stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def bm25_idf_command(term: str) -> float:
    stopwords_path = Path(__file__).parent.parent / "data" / "stopwords.txt"
    with open(stopwords_path, "r") as f:
        stopwords = f.read().splitlines()

    stemmer = PorterStemmer()
    idx = InvertedIndex(stopwords, stemmer)
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    stopwords_path = Path(__file__).parent.parent / "data" / "stopwords.txt"
    with open(stopwords_path, "r") as f:
        stopwords = f.read().splitlines()

    stemmer = PorterStemmer()
    idx = InvertedIndex(stopwords, stemmer)
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to query")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to query")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to query")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
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
        case "tf":
            idx = InvertedIndex(stopwords, stemmer)
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            try:
                tf = idx.get_tf(args.doc_id, args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(tf)
        case "idf":
            idx = InvertedIndex(stopwords, stemmer)
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            term_tokens = normalize_text(args.term, stopwords, stemmer)
            if len(term_tokens) != 1:
                print("Error: term must tokenize to exactly one token")
                return

            token = term_tokens[0]
            total_doc_count = len(idx.docmap)
            term_match_doc_count = len(idx.index.get(token, set()))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "bm25idf":
            try:
                bm25_idf = bm25_idf_command(args.term)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            try:
                bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            except ValueError as e:
                print(f"Error: {e}")
                return

            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case "bm25search":
            idx = InvertedIndex(stopwords, stemmer)
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            results = idx.bm25_search(args.query, args.limit)
            for i, (doc_id, score) in enumerate(results, 1):
                title = idx.docmap[doc_id]["title"]
                print(f"{i}. ({doc_id}) {title} - BM25: {score:.2f}")
        case "tfidf":
            idx = InvertedIndex(stopwords, stemmer)
            try:
                idx.load()
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return

            term_tokens = normalize_text(args.term, stopwords, stemmer)
            if len(term_tokens) != 1:
                print("Error: term must tokenize to exactly one token")
                return

            try:
                tf = idx.get_tf(args.doc_id, args.term)
            except ValueError as e:
                print(f"Error: {e}")
                return

            docmap = idx.docmap
            total_doc_count = len(docmap)
            term_match_doc_count = len(idx.index.get(term_tokens[0], set()))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = (tf * idf)
            
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
