"""
Chuyển đổi dữ liệu đã trích xuất thành LangChain Document objects.
"""

import logging
from langchain_core.documents import Document
from backend.data_ingestion.schemas import CandidateProfile
from backend.data_ingestion.loading.loaders import batch_process_cvs, batch_process_company_docs
from typing import List

logger = logging.getLogger(__name__)


def profile_to_document(profile: CandidateProfile, file_path) -> Document:

    # 1. Skills
    skills_desc = ", ".join(profile.skills) if profile.skills else "N/A"
    skill_names = skills_desc

    # 2. Experience
    experience_text = "\n".join(profile.experiences) if profile.experiences else "N/A"

    # 3. Education
    education_text = "\n".join(profile.education) if profile.education else "N/A"

    # --- PAGE CONTENT ---
    page_content = f"""
        Candidate Name: {profile.full_name or "N/A"}
        Professional Summary: {profile.summary or "N/A"}

        Top Skills: {skills_desc}

        Work Experience:
        {experience_text}

        Education:
        {education_text}
        """.strip()

    # --- METADATA ---
    metadata = {
        "full_name": profile.full_name or "",
        "email": profile.email or "",
        "phone": profile.phone or "",
        "skills": skill_names,
        "file_name": file_path
    }

    return Document(
        page_content=page_content,
        metadata=metadata
    )

def get_cv_Docs(directory_path: str):
    """Load CVs, extract profiles, và chuyển thành LangChain Documents."""
    logger.info(f"Bắt đầu gọi batch_process_cvs từ thư mục: {directory_path}...")
    profiles, file_paths = batch_process_cvs(directory_path)
    logger.info(f"Hoàn tất batch_process_cvs. Đã trích xuất được {len(profiles)} profiles.")
    
    docs = [profile_to_document(profile, path) for profile, path in zip(profiles, file_paths)]
    logger.info("Hoàn tất chuyển đổi profile sang Document.")
    return docs

def get_company_Docs(directory_path: str):
    """Load company documents và chuyển thành LangChain Documents."""
    logger.info(f"Bắt đầu gọi batch_process_company_docs từ thư mục: {directory_path}...")
    company_texts, file_paths = batch_process_company_docs(directory_path)
    logger.info(f"Hoàn tất batch_process_company_docs. Lấy được {len(company_texts)} docs.")
    
    return [
        Document(page_content=text, metadata={"file_path": path}) 
        for text, path in zip(company_texts, file_paths)
    ]
