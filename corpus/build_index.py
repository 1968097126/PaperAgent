"""Build paper chunks and FAISS/BM25 indexes: python -m corpus.build_index."""

from corpus.indexing import main

if __name__ == "__main__":
    raise SystemExit(main())
