"""
Chịu trách nhiệm đọc file và xử lý văn bản (Load PDF, Chunk text)
"""

from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str) -> str:
    """Đọc và trích xuất nội dung từ file PDF."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    return full_text

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Chia nhỏ văn bản thành các đoạn (chunk) để LLM dễ xử lý."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks
