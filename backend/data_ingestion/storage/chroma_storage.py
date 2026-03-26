"""
ChromaDB Vector Store storage
"""

import os
import hashlib
import logging
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def _doc_id(text: str) -> str:
    """Tạo ID deterministic cho document (tránh trùng lặp)"""
    return hashlib.sha256(text.encode()).hexdigest()


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
        logger.warning(f"Không có dữ liệu cho {collection_name}. Bỏ qua.")
        return None, [], []

    # Chia nhỏ documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    ids = [f"{i}_{_doc_id(doc.page_content)}" for i, doc in enumerate(splits)]

    logger.info(f"Đã cắt thành {len(splits)} chunks cho '{collection_name}'. Đang tạo embeddings...")

    # Tạo embeddings
    texts = [doc.page_content for doc in splits]
    doc_embeddings = embedding_model.embed_documents(texts)

    logger.info(f"Embedding xong! Đang lưu vào ChromaDB...")

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

    logger.info(f"Lưu ChromaDB '{collection_name}' hoàn tất.")
    return vectorstore, doc_embeddings


def load_chromadb(collection_name: str, embedding_model, chroma_dir: str) -> Chroma:
    """Load một Chroma collection đã tồn tại"""
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
        embedding_function=embedding_model,
    )
