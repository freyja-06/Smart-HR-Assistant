from concurrent.futures import ThreadPoolExecutor, as_completed
from backend.data_ingestion.extractors import process_single_cv, load_pdf
from typing import List
import glob
import os
import logging


logger = logging.getLogger(__name__)

def batch_process_cvs(directory_path: str):
    """
    Scan the entire directory and process all detected PDF files.
    Return a list of successfully extracted CandidateProfile objects and their file paths.
    """

    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    results = []
    successful_file_paths = [] 
    max_workers = 4 

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Mapping future -> file path
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

                print(f"[{idx}/{len(pdf_files)}] Done")

            except Exception as e:
                print(f"Lỗi file {pdf_path}: {e}")

    print(f"Hoàn tất: {len(results)}/{len(pdf_files)}")

    # Trả về cả list kết quả và list đường dẫn file
    return results, successful_file_paths

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