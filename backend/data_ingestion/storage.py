"""
Unified Storage Module
======================
Module thống nhất tất cả thao tác lưu/đọc dữ liệu:
  - LangChain Documents (.pkl)
  - Embeddings (.npy)
  - ChromaDB (vector store)
  - BM25 index (.pkl)
"""

import os
import pickle
import hashlib
import numpy as np
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi


# =====================================================================
#  1. LangChain Documents (.pkl)
# =====================================================================

def save_langdocs(docs: List[Document], folder_path: str, var_name: str) -> None:
    """Lưu danh sách LangChain Documents xuống file .pkl"""
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{var_name}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"💾 Đã lưu {var_name} tại: {file_path}")


def load_langdocs(folder_path: str, var_name: str):
    """Load danh sách LangChain Documents từ file .pkl"""
    file_path = os.path.join(folder_path, f"{var_name}.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"📂 Đã load {var_name} từ: {file_path}")
    return data


# =====================================================================
#  2. Embeddings (.npy)
# =====================================================================

def save_embeddings(embeddings, file_path: str) -> None:
    """Lưu embeddings xuống file .npy"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, embeddings)
    print(f"💾 Đã lưu embeddings tại: {file_path}")


def load_embeddings(file_path: str) -> np.ndarray:
    """Load embeddings từ file .npy"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    print(f"📂 Đã load embeddings từ: {file_path}")
    return data


# =====================================================================
#  3. ChromaDB (Vector Store)
# =====================================================================

def _doc_id(text: str) -> str:
    """Tạo ID deterministic cho document (tránh trùng lặp)"""
    return hashlib.md5(text.encode()).hexdigest()


def save_to_chromadb(
    docs: List[Document],
    collection_name: str,
    embedding_model,
    chroma_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Chia nhỏ documents, tạo embeddings, và lưu vào ChromaDB.
    
    Returns:
        (vectorstore, doc_embeddings, splits)
    """
    if not docs:
        print(f"Không có dữ liệu cho {collection_name}. Bỏ qua.")
        return None, [], []

    # Chia nhỏ documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    ids = [f"{i}_{_doc_id(doc.page_content)}" for i, doc in enumerate(splits)]

    print(f"[Storage] Đã cắt thành {len(splits)} chunks cho '{collection_name}'. Đang tạo embeddings...")

    # Tạo embeddings
    texts = [doc.page_content for doc in splits]
    doc_embeddings = embedding_model.embed_documents(texts)

    print(f"[Storage] Embedding xong! Đang lưu vào ChromaDB...")

    # Lưu vào ChromaDB
    chroma_dir_str = str(chroma_dir)
    if os.path.exists(chroma_dir_str):
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=chroma_dir_str,
            embedding_function=embedding_model,
        )
        vectorstore.add_documents(documents=splits, ids=ids)
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=chroma_dir_str,
            collection_name=collection_name,
        )

    print(f"[Storage] Lưu ChromaDB '{collection_name}' hoàn tất.")
    return vectorstore, doc_embeddings, splits


def load_chromadb(collection_name: str, embedding_model, chroma_dir: str) -> Chroma:
    """Load một Chroma collection đã tồn tại"""
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
        embedding_function=embedding_model,
    )


# =====================================================================
#  4. BM25 Index (.pkl)
# =====================================================================

def save_bm25_index(documents: List[Document], save_path: str) -> dict:
    """
    Tạo BM25 index từ documents và lưu xuống file .pkl.

    Returns:
        dict chứa 'bm25' và 'corpus'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tokenized_docs = [word_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    data = {"bm25": bm25, "corpus": tokenized_docs}

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Đã lưu BM25 index vào {save_path}")
    return data


def load_bm25_index(filepath: str):
    """
    Load BM25 index từ file .pkl.

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


# =====================================================================
#  5. Unified save_all / load_all
# =====================================================================

def save_all(
    docs: List[Document],
    collection_name: str,
    embedding_model,
    chroma_dir: str,
    langdocs_dir: str,
    langdocs_var_name: str,
    embedding_save_path: str,
    bm25_save_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Thực hiện toàn bộ quy trình lưu trữ:
      1. Lưu LangChain Documents (.pkl)
      2. Tạo embeddings + lưu vào ChromaDB
      3. Lưu embeddings (.npy)
      4. Tạo + lưu BM25 index (.pkl)

    Returns:
        (vectorstore, doc_embeddings)
    """
    # 1. Lưu LangDocs
    save_langdocs(docs, langdocs_dir, langdocs_var_name)

    # 2. ChromaDB (bao gồm tạo embeddings)
    vectorstore, doc_embeddings, splits = save_to_chromadb(
        docs=docs,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chroma_dir=chroma_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3. Lưu embeddings
    if doc_embeddings:
        save_embeddings(doc_embeddings, embedding_save_path)

    # 4. Lưu BM25
    save_bm25_index(docs, bm25_save_path)

    return vectorstore, doc_embeddings
