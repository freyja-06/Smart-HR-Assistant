"""
CVLoader — Load và chuyển đổi CV PDF thành LangChain Documents.

Logic riêng biệt cho CV được đóng gói thành private methods:
  - _batch_extract_profiles(): Dùng ThreadPool + LLM để extract song song
  - _profile_to_document(): Chuyển CandidateProfile → Document có cấu trúc

Shared utilities (load_pdf, chunk_text, extract_chunk, merge_profiles)
được import từ bên ngoài để tái sử dụng.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from backend.data_ingestion.loading.base_loader import BaseLoader
from backend.data_ingestion.extraction.cv_processor import process_single_cv
from backend.data_ingestion.schemas import CandidateProfile
from typing import List, Tuple
import glob
import os
import logging

logger = logging.getLogger(__name__)


class CVLoader(BaseLoader):
    """Loader chuyên xử lý CV ứng viên: PDF → LLM Extract → Document."""

    def __init__(self, max_workers: int = 4):
        self._max_workers = max_workers

    # ─── Template Method implementations ─────────────────────

    def _load_raw(self, directory_path: str) -> Tuple[List[CandidateProfile], List[str]] | None:
        """Load tất cả PDF CV, extract thông tin bằng LLM song song."""
        profiles, file_paths = self.__batch_extract_profiles(directory_path)
        if not profiles:
            return None
        return profiles, file_paths

    def _transform(self, raw_data) -> List[Document]:
        """Chuyển danh sách CandidateProfile → LangChain Documents."""
        profiles, file_paths = raw_data
        return [
            self.__profile_to_document(profile, path)
            for profile, path in zip(profiles, file_paths)
        ]

    # ─── Private methods (logic riêng của CV) ────────────────

    def __batch_extract_profiles(
        self, directory_path: str
    ) -> Tuple[List[CandidateProfile], List[str]]:
        """
        Quét thư mục, extract thông tin từ từng CV bằng LLM (song song).

        Returns:
            (profiles, file_paths) — danh sách profile và đường dẫn file tương ứng.
        """
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files:
            logger.warning(f"Không tìm thấy file PDF trong: {directory_path}")
            return [], []

        results = []
        successful_file_paths = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(process_single_cv, pdf): pdf
                for pdf in pdf_files
            }

            for idx, future in enumerate(as_completed(futures), start=1):
                pdf_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        successful_file_paths.append(pdf_path)
                    logger.info(f"[CV {idx}/{len(pdf_files)}] Done")
                except Exception as e:
                    logger.error(f"Lỗi file {pdf_path}: {e}")

        logger.info(f"Extract hoàn tất: {len(results)}/{len(pdf_files)} CVs")
        return results, successful_file_paths

    def __profile_to_document(
        self, profile: CandidateProfile, file_path: str
    ) -> Document:
        """
        Chuyển một CandidateProfile thành LangChain Document.

        Page content chứa thông tin có cấu trúc để embedding.
        Metadata chứa các trường dùng cho filtering.
        """
        # Build text blocks
        skills_desc = ", ".join(profile.skills) if profile.skills else "N/A"
        experience_text = "\n".join(profile.experiences) if profile.experiences else "N/A"
        education_text = "\n".join(profile.education) if profile.education else "N/A"

        page_content = f"""
        Candidate Name: {profile.full_name or "N/A"}
        Professional Summary: {profile.summary or "N/A"}

        Top Skills: {skills_desc}

        Work Experience:
        {experience_text}

        Education:
        {education_text}
        """.strip()

        metadata = {
            "full_name": profile.full_name or "",
            "email": profile.email or "",
            "phone": profile.phone or "",
            "skills": skills_desc,
            "file_name": file_path,
        }

        return Document(page_content=page_content, metadata=metadata)
