import pickle
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any


class InvertedIndex:
    def __init__(self, stopwords=None, stemmer=None):
        self.index = defaultdict(set)
        self.docmap = {}
        self.stopwords = stopwords or []
        self.stemmer = stemmer
        self.term_freqs = Counter()
    
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.__tokenize(text)
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

    def load(self) -> None:
        cache_dir = Path("cache")
        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"
        term_freqs_path = cache_dir / "term_freqs.pkl"
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
        