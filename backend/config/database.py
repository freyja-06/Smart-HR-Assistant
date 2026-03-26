"""
Database Configuration
======================
Module kết nối vào các database đã tồn tại (ChromaDB, BM25, LangDocs, Embeddings).
Chỉ dùng cho RUNTIME (không dùng cho ingest).
"""

from dotenv import load_dotenv
import backend.constant_variables as const
from backend.data_ingestion.storage import (
    load_chromadb,
    load_bm25_index,
    load_langdocs,
    load_embeddings,
)
from backend.agents.llm_processor.llm_factory import ModelFactory

load_dotenv()

# =====================================================================
#  Embedding Model (dùng chung cho tất cả Chroma collections)
# =====================================================================

EMBEDDING = ModelFactory.create(
    model_type="embedding",
    provider="ollama",
    model_name="nomic-embed-text",
)

# =====================================================================
#  ChromaDB Vector Stores
# =====================================================================

cv_store = load_chromadb(
    collection_name="CVs",
    embedding_model=EMBEDDING,
    chroma_dir=str(const.CHROMA_DIR),
)

company_docs_store = load_chromadb(
    collection_name="company_docs",
    embedding_model=EMBEDDING,
    chroma_dir=str(const.CHROMA_DIR),
)

# =====================================================================
#  BM25 Indexes
# =====================================================================

cv_bm25, cv_corpus = load_bm25_index(str(const.BM25_CV_PATH))
company_docs_bm25, company_docs_corpus = load_bm25_index(str(const.BM25_COMPANY_PATH))

# =====================================================================
#  LangChain Documents & Embeddings
# =====================================================================

cv_docs = load_langdocs(
    folder_path=str(const.LANGDOCS_SAVE_DIR),
    var_name="cv_docs",
)

company_docs = load_langdocs(
    folder_path=str(const.LANGDOCS_SAVE_DIR),
    var_name="company_docs",
)

company_embeddings = load_embeddings(str(const.COMPANY_EMBEDDING_SAVE_DIR))
cv_embeddings = load_embeddings(str(const.CV_EMBEDDING_SAVE_DIR))

# =====================================================================
#  Public API
# =====================================================================

__all__ = [
    "cv_docs",
    "company_docs",
    "cv_store",
    "company_docs_store",
    "cv_bm25",
    "company_docs_bm25",
    "cv_embeddings",
    "company_embeddings",
    "EMBEDDING",
]