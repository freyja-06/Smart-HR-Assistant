"""
Loading package: PDF loading and text processing utilities.

Không re-export eagerly để tránh circular import.
Import trực tiếp từ submodule khi cần:
  from backend.data_ingestion.loading.text_utils import load_pdf, chunk_text
  from backend.data_ingestion.loading.loaders import batch_process_cvs, batch_process_company_docs
"""
