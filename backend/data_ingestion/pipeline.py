"""
Ingest Pipeline Orchestrator
=============================
Điều phối toàn bộ quy trình nạp dữ liệu:
  Load PDF → Transform → Save All (ChromaDB + Embeddings + BM25 + LangDocs)
"""

from dotenv import load_dotenv
from backend.data_ingestion.transformation.transformers_LangDocs import get_cv_Docs, get_company_Docs
from backend.data_ingestion.storage import save_all
from backend.agents.llm_processor.llm_factory import ModelFactory
import backend.constant_variables as const

load_dotenv()


def _get_embedding_model():
    """Khởi tạo embedding model dùng chung cho pipeline"""
    return ModelFactory.create(
        model_type="embedding",
        provider="ollama",
        model_name="nomic-embed-text",
    )


def run_cv_pipeline(
    cv_path: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Pipeline nạp dữ liệu CV:
      1. Load & extract thông tin từ PDF
      2. Transform thành LangChain Documents
      3. Lưu tất cả (ChromaDB, embeddings, BM25, langdocs)
    """
    cv_path = cv_path or str(const.CV_PATH)

    print("\n[Pipeline] Đang xử lý CVs...")
    print(f"[Pipeline] Đường dẫn: {cv_path}")

    # 1. Load & Transform
    cv_docs = get_cv_Docs(cv_path)

    if not cv_docs:
        print("[Pipeline] Không tìm thấy CV nào. Bỏ qua.")
        return

    # 2. Save All
    embedding_model = _get_embedding_model()

    save_all(
        docs=cv_docs,
        collection_name="CVs",
        embedding_model=embedding_model,
        chroma_dir=str(const.CHROMA_DIR),
        langdocs_dir=str(const.LANGDOCS_SAVE_DIR),
        langdocs_var_name="cv_docs",
        embedding_save_path=str(const.CV_EMBEDDING_SAVE_DIR),
        bm25_save_path=str(const.BM25_CV_PATH),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("[Pipeline] Hoàn tất nạp dữ liệu CV")


def run_company_docs_pipeline(
    company_path: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Pipeline nạp dữ liệu company documents:
      1. Load PDF company documents
      2. Transform thành LangChain Documents
      3. Lưu tất cả (ChromaDB, embeddings, BM25, langdocs)
    """
    company_path = company_path or str(const.COMPANY_DOCS_PATH)

    print("\n[Pipeline] Đang xử lý Company Documents...")
    print(f"[Pipeline] Đường dẫn: {company_path}")

    # 1. Load & Transform
    company_docs = get_company_Docs(company_path)

    if not company_docs:
        print("[Pipeline] Không tìm thấy company documents nào. Bỏ qua.")
        return

    # 2. Save All
    embedding_model = _get_embedding_model()

    save_all(
        docs=company_docs,
        collection_name="company_docs",
        embedding_model=embedding_model,
        chroma_dir=str(const.CHROMA_DIR),
        langdocs_dir=str(const.LANGDOCS_SAVE_DIR),
        langdocs_var_name="company_docs",
        embedding_save_path=str(const.COMPANY_EMBEDDING_SAVE_DIR),
        bm25_save_path=str(const.BM25_COMPANY_PATH),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("[Pipeline] ✅ Hoàn tất nạp dữ liệu Company Documents")


def run_full_pipeline():
    """Chạy toàn bộ pipeline nạp dữ liệu"""
    print("=" * 50)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH NẠP DỮ LIỆU VÀO VECTOR DATABASE")
    print("=" * 50)

    try:
        print("\n[1/2] Đang xử lý thư mục chứa CVs...")
        run_cv_pipeline()

        print("\n[2/2] Đang xử lý thư mục chứa Company Documents...")
        run_company_docs_pipeline()

        print("\n✅ XONG! Tất cả dữ liệu đã được lưu an toàn.")

    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi trong quá trình nạp dữ liệu: {e}")
        raise
