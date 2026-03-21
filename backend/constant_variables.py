from pathlib import Path
BASE_DIR = Path(__file__).resolve()


# Path
CV_PATH: str = BASE_DIR.parent / "data" / "All_pdf_cv"
COMPANY_DOCS_PATH: str = BASE_DIR.parent / "data" / "Company_documents"

BM25_CV_PATH = BASE_DIR / "database" / "all_bm25_index" / "cv_bm25.pkl"
BM25_COMPANY_PATH = BASE_DIR / "database" / "all_bm25_index" / "company_bm25.pkl"


LANGDOCS_SAVE_DIR = BASE_DIR / "database" / "all_langDocs"

EMBEDDING_SAVE_DIR = BASE_DIR / "database" / "all_embeddings"

CV_EMBEDDING_SAVE_DIR = EMBEDDING_SAVE_DIR / "cv_embeddings.npy"
COMPANY_EMBEDDING_SAVE_DIR = EMBEDDING_SAVE_DIR / "company_embeddings.npy"

CHROMA_DIR = BASE_DIR / "database" / "chroma_db"
