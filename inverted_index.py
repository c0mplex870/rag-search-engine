import os
import pickle
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any
import math
from constants import BM25_K1, BM25_B, CACHE_DIR

class InvertedIndex:
    def __init__(self, stopwords=None, stemmer=None):
        self.index = defaultdict(set)
        self.docmap = {}
        self.stopwords = stopwords or []
        self.stemmer = stemmer
        self.term_freqs = Counter()
        self.doc_lengths = {}
    def _get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
       
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.__tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_freqs[token] += 1

    def __tokenize(self, text: str) -> list[str]:
        import string
        translator = str.maketrans('', '', string.punctuation)
        clean_text = text.translate(translator).lower()
        tokens = [t for t in clean_text.split() if t and t not in self.stopwords]
        if self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term.lower(), set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        term_tokens = self.__tokenize(term)
        if len(term_tokens) != 1:
            raise ValueError("term must tokenize to exactly one token")

        movie = self.docmap.get(doc_id)
        if not movie:
            return 0

        token = term_tokens[0]
        text = f"{movie['title']} {movie['description']}"
        doc_tokens = self.__tokenize(text)
        return doc_tokens.count(token)

    def get_bm25_idf(self, term: str) -> float:
        term_tokens = self.__tokenize(term)
        if len(term_tokens) != 1:
            raise ValueError("term must tokenize to exactly one token")

        token = term_tokens[0]
        total_doc_count = len(self.docmap)
        doc_freq = len(self.index.get(token, set()))
        return math.log((total_doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1 = BM25_K1, b = BM25_B) -> float:
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self._get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length else 0

        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm) if length_norm > 0 else 0.0
    
    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    def bm25_search(self, query, limit):
        query_tokens = self.__tokenize(query)
        scores = {}

        for doc_id in self.docmap:
            total_score = 0.0
            for token in query_tokens:
                total_score += self.bm25(doc_id, token)
            scores[doc_id] = total_score

        ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_scores[:limit]
     
            
    def build(self, movies: list[dict[str, Any]]) -> None:
        for m in movies:
            doc_id = int(m["id"])
            self.docmap[doc_id] = m
            text = f"{m['title']} {m['description']}"
            self.__add_document(doc_id, text)

    def save(self) -> None:
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache_dir / "index.pkl", "wb") as f:
            pickle.dump(dict(self.index), f)

        with open(cache_dir / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open(cache_dir / "term_freqs.pkl", "wb") as f:
            pickle.dump(self.term_freqs, f)
        with open(cache_dir / "doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        cache_dir = Path("cache")
        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"
        term_freqs_path = cache_dir / "term_freqs.pkl"
        doc_lengths_path = cache_dir / "doc_lengths.pkl"

        if not index_path.exists() or not docmap_path.exists():
            raise FileNotFoundError("Cache files not found. Please run 'build' first.")

        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
            self.index = defaultdict(set)
            for token, doc_ids in index_data.items():
                self.index[token] = set(doc_ids)

        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        if term_freqs_path.exists():
            with open(term_freqs_path, "rb") as f:
                self.term_freqs = pickle.load(f)
        if doc_lengths_path.exists():
            with open(doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)

    