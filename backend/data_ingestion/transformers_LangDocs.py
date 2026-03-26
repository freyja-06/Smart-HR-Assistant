from langchain_core.documents import Document
from backend.data_ingestion.schemas import CandidateProfile
from backend.data_ingestion.loaders import batch_process_cvs, batch_process_company_docs
from typing import List


def profile_to_document(profile: CandidateProfile) -> Document:

    # 1. Skills
    skills_desc = ", ".join(profile.skills) if profile.skills else "N/A"
    skill_names = skills_desc  # đã là string rồi

    # 2. Experience
    experience_text = "\n".join(profile.experiences) if profile.experiences else "N/A"

    # 3. Education
    education_text = "\n".join(profile.education) if profile.education else "N/A"

    # 4. Languages (bạn đã bỏ field này → remove luôn hoặc optional)
    langs = "N/A"

    # --- PAGE CONTENT ---
    page_content = f"""
        Candidate Name: {profile.full_name}
        Professional Summary: {profile.summary or "N/A"}

        Top Skills: {skills_desc}

        Work Experience:
        {experience_text}

        Education:
        {education_text}
        """.strip()

    # --- METADATA ---
    metadata = {
        "full_name": profile.full_name,
        "email": profile.email,
        "phone": profile.phone or "",
        "skills": skill_names,
        "file_name": profile.cv_file_name
    }

    return Document(
        page_content=page_content,
        metadata=metadata
    )

def get_cv_Docs(directory_path: str):
    print(f"\n[DEBUG] 1. Bắt đầu gọi batch_process_cvs từ thư mục: {directory_path}...")
    profiles: List[CandidateProfile] = batch_process_cvs(directory_path)
    print(f"[DEBUG] 1.1. Hoàn tất batch_process_cvs. Đã trích xuất được {len(profiles)} profiles.")
    
    docs = [profile_to_document(i) for i in profiles]
    print("[DEBUG] 1.2. Hoàn tất chuyển đổi profile sang Document.")
    return docs

def get_company_Docs(directory_path: str):
    print(f"\n[DEBUG] 2. Bắt đầu gọi batch_process_company_docs từ thư mục: {directory_path}...")
    company_texts, file_paths = batch_process_company_docs(directory_path)
    print(f"[DEBUG] 2.1. Hoàn tất batch_process_company_docs. Lấy được {len(company_texts)} docs.")
    
    company_docs = [Document(page_content=text) for text in company_texts]
    return [Document(page_content=company_docs[i].page_content, metadata = {"file_path": file_paths[i]}) for i in range(len(company_docs))]

