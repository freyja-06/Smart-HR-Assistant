from langchain_core.documents import Document
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from typing import List
import pickle
import os

def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    BM25 (Best Matching 25) is a ranking function used in information retrieval.
    It's based on the probabilistic retrieval framework and is an improvement over TF-IDF.

    Args:
    documents (List[Document]): List of documents to index.

    Returns:
    BM25Okapi: An index that can be used for BM25 scoring.
    """
    # Tokenize each document by splitting on whitespace
    # This is a simple approach and could be improved with more sophisticated tokenization
    tokenized_docs = [word_tokenize(doc.page_content) for doc in documents]
    return {
        "bm25": BM25Okapi(tokenized_docs),
        "corpus": tokenized_docs
    }
    
def save_bm25(bm25, corpus, filepath):
    """
    Lưu BM25 index và corpus ra file

    Args:
        bm25: BM25Okapi object
        corpus: list[str]
        filepath: đường dẫn file lưu
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "corpus": corpus
            }, f)
        print(f"✅ Đã lưu BM25 vào {filepath}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu BM25: {e}")

def load_bm25(filepath):
    """
    Load BM25 index từ file

    Args:
        filepath: đường dẫn file

    Returns:
        (bm25, corpus) hoặc (None, None) nếu lỗi
    """
    if not os.path.exists(filepath):
        print(f"❌ File {filepath} không tồn tại")
        return None, None

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        bm25 = data["bm25"]
        corpus = data["corpus"]

        print(f"✅ Đã load BM25 từ {filepath}")
        return bm25, corpus

    except Exception as e:
        print(f"❌ Lỗi khi load BM25: {e}")
        return None, None