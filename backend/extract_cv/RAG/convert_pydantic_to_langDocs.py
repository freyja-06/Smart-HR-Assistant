from langchain_core.documents import Document
from backend.extract_cv.candidate_profile import CandidateProfile
from backend.extract_cv.extract_pdf import batch_process_cvs, batch_process_company_docs
from typing import List


def profile_to_document(profile: CandidateProfile) -> Document:

# Xử lý các dữ liệu riêng lẻ thành dạng xâu

    # 1. Xử lý Skills:
    # Tạo chuỗi text để AI đọc (VD: "Python (Expert), Java (Intermediate)")
    skills_desc = ", ".join([f"{s.name} ({s.level.value})" for s in profile.skills]) if profile.skills else "N/A"
    # Chroma không cho phép metadata là kiểu list, dict, object, ..v.v
    skill_names = [s.name for s in profile.skills] if profile.skills else []
    skill_names = ", ".join(skill_names)

    # 2. Xử lý Experience:
    # Ghép các kinh nghiệm làm việc thành một đoạn văn mô tả
    experience_text = ""
    if profile.experience:
        for exp in profile.experience:
            start = exp.start_date
            end = exp.end_date if exp.end_date else "Present"
            experience_text += f"- {exp.job_title} at {exp.company} ({start} to {end}): {exp.description}\n"
    
    # 3. Xử lý Education:
    education_text = ""
    if profile.education:
        for edu in profile.education:
            education_text += f"- {edu.degree} in {edu.major} at {edu.institution}\n"

    # 4. Xử lý Languages:
    langs = ", ".join(profile.languages) if profile.languages else "N/A"

# --- TẠO PAGE CONTENT (Nội dung để Embedding) ---
    # Sắp xếp các thông tin quan trọng nhất lên đầu để Vector Search hiệu quả hơn
    page_content = f"""
    Candidate Name: {profile.full_name}
    Professional Summary: {profile.summary}
    
    Top Skills: {skills_desc}
    Languages: {langs}
    
    Work Experience:
    {experience_text}
    
    Education:
    {education_text}
    """.strip()

    # --- TẠO METADATA (Dữ liệu để lọc - Filter) ---
    # Lưu ý: Không lưu nested object vào metadata để tránh lỗi Vector DB
    metadata = {
        "full_name": profile.full_name,
        "email": profile.email,
        "phone": profile.phone if profile.phone else "",
        "address": profile.address if profile.address else "",
        "skills": skill_names,  # List[str] -> Dễ dàng filter: skills contains "Python"
        "languages": langs,
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

