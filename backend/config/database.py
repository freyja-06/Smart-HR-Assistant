from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from extract_cv.RAG.rag_backend import CHROMA_DIR

load_dotenv()

# 1. Khởi tạo embedding giống hệt như bên rag_backend.py
EMBEDDING = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 3. KẾT NỐI VÀO DATABASE ĐÃ TỒN TẠI
cv_store = Chroma(
    collection_name="CVs", 
    persist_directory=str(CHROMA_DIR),
    embedding_function=EMBEDDING
)

policies_store = Chroma(
    collection_name="Policies", 
    persist_directory=str(CHROMA_DIR),
    embedding_function=EMBEDDING
)