"""
Orchestrator cấp đơn CV: gọi các module loading, extraction, merging
để thực hiện luồng xử lý một file CV hoàn chỉnh.
"""

import logging
from backend.data_ingestion.loading.text_utils import load_pdf, chunk_text
from backend.data_ingestion.extraction.extractors import extract_chunk
from backend.data_ingestion.extraction.profile_merger import merge_profiles
from backend.data_ingestion.schemas import CandidateProfile

logger = logging.getLogger(__name__)

def process_single_cv(pdf_file: str) -> CandidateProfile | None:
    """
    Luồng xử lý một CV hoàn chỉnh: 
    Load -> Giới hạn dung lượng -> Chunking -> LLM Extract -> Merging.
    """
    try:
        # 1. Load document
        cv_text = load_pdf(pdf_file)

        # 2. Limit size để tránh overload LLM
        cv_text = cv_text[:12000]

        # 3. Text chunking
        chunks = chunk_text(cv_text)

        # 4. Extract data using LLM
        results = []
        for chunk in chunks:
            res = extract_chunk(chunk)
            if res:
                results.append(res)

        # 5. Merge and return
        return merge_profiles(results)

    except Exception as e:
        logger.error(f"Lỗi khi xử lý file {pdf_file}: {e}")
        return None
