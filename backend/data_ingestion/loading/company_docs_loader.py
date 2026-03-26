"""
CompanyDocsLoader — Load và chuyển đổi tài liệu công ty thành LangChain Documents.

Logic riêng biệt cho company docs được đóng gói thành private methods:
  - _load_all_pdfs(): Đọc tất cả PDF từ thư mục

Shared utilities (load_pdf) được import từ bên ngoài để tái sử dụng.
"""

from langchain_core.documents import Document
from backend.data_ingestion.loading.base_loader import BaseLoader
from backend.data_ingestion.loading.text_utils import load_pdf
from typing import List, Tuple
import glob
import os
import logging

logger = logging.getLogger(__name__)


class CompanyDocsLoader(BaseLoader):
    """Loader chuyên xử lý tài liệu công ty: PDF → Document đơn giản."""

    # ─── Template Method implementations ─────────────────────

    def _load_raw(self, directory_path: str) -> Tuple[List[str], List[str]] | None:
        """Load tất cả PDF company docs, trả về nội dung text và đường dẫn."""
        texts, file_paths = self.__load_all_pdfs(directory_path)
        if not texts:
            return None
        return texts, file_paths

    def _transform(self, raw_data) -> List[Document]:
        """Chuyển danh sách text → LangChain Documents với metadata file_path."""
        texts, file_paths = raw_data
        return [
            Document(page_content=text, metadata={"file_path": path})
            for text, path in zip(texts, file_paths)
        ]

    # ─── Private methods (logic riêng của Company Docs) ──────

    def __load_all_pdfs(
        self, directory_path: str
    ) -> Tuple[List[str], List[str]]:
        """
        Đọc tất cả file PDF trong thư mục và trả về nội dung text.

        Returns:
            (texts, file_paths) — nội dung và đường dẫn file tương ứng.
        """
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files:
            logger.warning(f"Không tìm thấy file PDF trong: {directory_path}")
            return [], []

        processed_texts: List[str] = []
        successful_paths: List[str] = []

        for pdf_file in pdf_files:
            try:
                text = load_pdf(pdf_file)
                if text:
                    processed_texts.append(text)
                    successful_paths.append(pdf_file)
            except Exception as e:
                logger.error(f"Lỗi khi đọc file {pdf_file}: {e}")

        logger.info(f"Load hoàn tất: {len(processed_texts)}/{len(pdf_files)} documents")
        return processed_texts, successful_paths
