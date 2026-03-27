"""
Unified save_all: điều phối lưu trữ vào tất cả các backend cùng lúc.
"""

import logging
from typing import List
from langchain_core.documents import Document

from backend.data_ingestion.storage.langdocs_storage import save_langdocs
from backend.data_ingestion.storage.embedding_storage import save_embeddings
from backend.data_ingestion.storage.chroma_storage import save_to_chromadb
from backend.data_ingestion.storage.bm25_storage import save_bm25_index

logger = logging.getLogger(__name__)


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


    Cấu trúc lưu file này chưa được tối ưu!
    """
    # 1. Lưu LangDocs
    save_langdocs(docs, langdocs_dir, langdocs_var_name)

    # 2. ChromaDB (bao gồm tạo embeddings)
    vectorstore, doc_embeddings = save_to_chromadb(
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
