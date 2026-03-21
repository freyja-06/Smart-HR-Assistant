from langchain_chroma import Chroma
from dotenv import load_dotenv
import backend.constant_variables as const
from extract_cv.RAG.bm25_module import load_bm25
from backend.config.loading_and_caching import load_or_create
import numpy as np
from backend.agents.llm_processor.llm_factory import ModelFactory

CHROMA_DIR = const.CHROMA_DIR
BM25_CV_PATH = const.BM25_CV_PATH 
BM25_COMPANY_PATH = const.BM25_COMPANY_PATH
LANGDOCS_SAVE_DIR = const.LANGDOCS_SAVE_DIR
CV_EMBEDDING_SAVE_DIR = const.CV_EMBEDDING_SAVE_DIR
COMPANY_EMBEDDING_SAVE_DIR = const.COMPANY_EMBEDDING_SAVE_DIR 

load_dotenv()

EMBEDDING = ModelFactory()

# KẾT NỐI VÀO DATABASE ĐÃ TỒN TẠI
cv_store = Chroma(
    collection_name="CVs", 
    persist_directory=str(CHROMA_DIR),
    embedding_function=EMBEDDING
)

company_docs_store = Chroma(
    collection_name="company_docs",
    persist_directory=str(CHROMA_DIR),
    embedding_function=EMBEDDING
)

cv_bm25, cv_corpus = load_bm25(BM25_CV_PATH)
company_docs_bm25, company_docs_corpus = load_bm25(BM25_COMPANY_PATH)

cv_docs = load_or_create(folder_path = LANGDOCS_SAVE_DIR , var_name = "cv_docs")
company_docs = load_or_create(folder_path = LANGDOCS_SAVE_DIR , var_name = "company_docs")


company_embeddings = np.load(COMPANY_EMBEDDING_SAVE_DIR, allow_pickle=True)
cv_embeddings = np.load(CV_EMBEDDING_SAVE_DIR, allow_pickle=True)

__all__ = [
    "cv_docs",
    "company_docs",
    "cv_store",
    "company_docs_store",
    "cv_bm25",
    "company_docs_bm25",
    "cv_embeddings",
    "company_embeddings"
]