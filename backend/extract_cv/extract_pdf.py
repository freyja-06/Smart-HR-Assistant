from langchain_community.document_loaders import PyPDFLoader
from backend.extract_cv.candidate_profile import CandidateProfile 
from backend.extract_cv.extract_to_pydantic_model import extract_cv_str_data
from typing import List
import glob
import os
import logging


logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    # Concatenate all contents
    full_text = "\n".join([page.page_content for page in pages]) + f"\n \n Đường dẫn tới tệp tài liệu pdf này: {file_path}"
    return full_text


def batch_process_cvs(directory_path: str):
    """
    Scan the entire directory and process all detected PDF files.
    Return a list of successfully extracted CandidateProfile objects.
    """

    processed_profiles: List[CandidateProfile] = []

    # Find all files .pdf in folder
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    for pdf_file in pdf_files:
        try:
            cv_text = load_pdf(pdf_file)
            if cv_text:
                processed_profiles.append(extract_cv_str_data(cv_text))

        except Exception as e:
            logger.error(f"Lỗi khi xử lý tệp {pdf_file}: {str(e)}", exc_info=True)

    return processed_profiles

def batch_process_company_docs(directory_path: str):
    processed_company_docs: List[str] = []
    file_paths: List[str] = []

    # Find all files .pdf in folder
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    for pdf_file in pdf_files:
        try:
            company_text = load_pdf(pdf_file)
            if company_text:
                processed_company_docs.append(company_text)
                file_paths.append(pdf_file)

        except Exception as e:
            print(f"File handling error {pdf_file}: {e}")

    return processed_company_docs, file_paths