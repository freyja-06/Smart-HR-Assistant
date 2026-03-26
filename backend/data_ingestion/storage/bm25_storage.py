"""
BM25 Index storage (.pkl)
Gộp từ bm25_indexers.py và phần BM25 trong storage.py — loại bỏ code trùng lặp.
"""

import os
import pickle
import logging
from typing import List
from langchain_core.documents import Document
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def create_bm25_index(documents: List[Document]) -> dict:
    """
    Tạo BM25 index từ danh sách documents.

    Args:
        documents: List of LangChain Documents to index.

    Returns:
        dict chứa 'bm25' (BM25Okapi) và 'corpus' (tokenized docs).
    """
    tokenized_docs = [word_tokenize(doc.page_content) for doc in documents]
    return {
        "bm25": BM25Okapi(tokenized_docs),
        "corpus": tokenized_docs
    }


def save_bm25_index(documents: List[Document], save_path: str) -> dict:
    """
    Tạo BM25 index từ documents và lưu xuống file .pkl.

    Returns:
        dict chứa 'bm25' và 'corpus'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = create_bm25_index(documents)

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Đã lưu BM25 index vào {save_path}")
    return data


def load_bm25_index(filepath: str):
    """
    Load BM25 index từ file .pkl.

    Returns:
        (bm25, corpus) hoặc (None, None) nếu lỗi
    """
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} không tồn tại")
        return None, None

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        bm25 = data["bm25"]
        corpus = data["corpus"]

        logger.info(f"Đã load BM25 từ {filepath}")
        return bm25, corpus

    except Exception as e:
        logger.error(f"Lỗi khi load BM25: {e}")
        return None, None
