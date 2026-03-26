"""
Ingest Pipeline Orchestrator
=============================
Điều phối toàn bộ quy trình nạp dữ liệu:
  Loader.get_docs(path) → Save All (ChromaDB + Embeddings + BM25 + LangDocs)

Sử dụng BaseLoader pattern (OCP):
  - Thêm nguồn dữ liệu mới → tạo subclass BaseLoader + thêm vào PIPELINE_CONFIGS
  - Không cần sửa code pipeline
"""

import logging
from dotenv import load_dotenv
from backend.data_ingestion.loading.base_loader import BaseLoader
from backend.data_ingestion.loading.cv_loader import CVLoader
from backend.data_ingestion.loading.company_docs_loader import CompanyDocsLoader
from backend.data_ingestion.storage import save_all
from backend.agents.llm_processor.llm_factory import ModelFactory
import backend.constant_variables as const

load_dotenv()
logger = logging.getLogger(__name__)


def _get_embedding_model():
    """Khởi tạo embedding model dùng chung cho pipeline"""
    return ModelFactory.create(
        model_type="embedding",
        provider="ollama",
        model_name="nomic-embed-text",
    )


# =====================================================================
#  Pipeline Configurations — Thêm nguồn dữ liệu mới? Thêm dict vào đây!
# =====================================================================

PIPELINE_CONFIGS = [
    {
        "name": "CVs",
        "loader": CVLoader(max_workers=4),
        "path": const.CV_PATH,
        "collection_name": "CVs",
        "langdocs_var_name": "cv_docs",
        "embedding_save_path": const.CV_EMBEDDING_SAVE_DIR,
        "bm25_save_path": const.BM25_CV_PATH,
    },
    {
        "name": "Company Documents",
        "loader": CompanyDocsLoader(),
        "path": const.COMPANY_DOCS_PATH,
        "collection_name": "company_docs",
        "langdocs_var_name": "company_docs",
        "embedding_save_path": const.COMPANY_EMBEDDING_SAVE_DIR,
        "bm25_save_path": const.BM25_COMPANY_PATH,
    },
    # Có thể thêm nguồn mới ở đây:
    # {
    #     "name": "Job Descriptions",
    #     "loader": JDLoader(),
    #     "path": const.JD_PATH,
    #     "collection_name": "job_descriptions",
    #     ...
    # },
]


def run_pipeline(
    loader: BaseLoader,
    path: str,
    collection_name: str,
    langdocs_var_name: str,
    embedding_save_path: str,
    bm25_save_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Pipeline generic: Nhận một BaseLoader bất kỳ và thực hiện:
      1. loader.get_docs(path) → LangChain Documents
      2. save_all() → Lưu vào ChromaDB + Embeddings + BM25 + LangDocs
    """
    docs = loader.get_docs(path)

    if not docs:
        logger.info(f"[Pipeline] Không có dữ liệu cho '{collection_name}'. Bỏ qua.")
        return

    embedding_model = _get_embedding_model()

    save_all(
        docs=docs,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chroma_dir=str(const.CHROMA_DIR),
        langdocs_dir=str(const.LANGDOCS_SAVE_DIR),
        langdocs_var_name=langdocs_var_name,
        embedding_save_path=str(embedding_save_path),
        bm25_save_path=str(bm25_save_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.info(f"[Pipeline] Hoàn tất nạp dữ liệu '{collection_name}'")


def run_full_pipeline():
    """Chạy toàn bộ pipeline nạp dữ liệu từ PIPELINE_CONFIGS."""
    print("=" * 50)
    print("BAT DAU QUA TRINH NAP DU LIEU VAO VECTOR DATABASE")
    print("=" * 50)

    try:
        total = len(PIPELINE_CONFIGS)
        for idx, config in enumerate(PIPELINE_CONFIGS, start=1):
            name = config["name"]
            print(f"\n[{idx}/{total}] Dang xu ly: {name}...")

            run_pipeline(
                loader=config["loader"],
                path=str(config["path"]),
                collection_name=config["collection_name"],
                langdocs_var_name=config["langdocs_var_name"],
                embedding_save_path=config["embedding_save_path"],
                bm25_save_path=config["bm25_save_path"],
            )

        print(f"\nXONG! Tat ca {total} nguon du lieu da duoc luu an toan.")

    except Exception as e:
        logger.error(f"Da xay ra loi trong qua trinh nap du lieu: {e}")
        raise
