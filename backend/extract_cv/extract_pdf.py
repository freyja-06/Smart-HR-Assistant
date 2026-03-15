from langchain_community.document_loaders import PyPDFLoader 
from extract_cv.candidate_profile import CandidateProfile 
from extract_cv.extract_to_pydantic_model import extract_cv_str_data
from typing import List
import glob
import os

def load_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    # Concatenate all contents
    full_text = "\n".join([page.page_content for page in pages])
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
            print(f"File handling error {pdf_file}: {e}")

    return processed_profiles

def batch_process_policies(directory_path: str):
    processed_policies: List[str] = []

    # Find all files .pdf in folder
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    for pdf_file in pdf_files:
        try:
            policy_text = load_pdf(pdf_file)
            if policy_text:
                processed_policies.append(policy_text)

        except Exception as e:
            print(f"File handling error {pdf_file}: {e}")

    return processed_policies