from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from extract_cv.RAG.rag_backend import CHROMA_DIR

load_dotenv()

# 1. Khởi tạo embedding giống hệt như bên rag_backend.py
# Hàm này không tốn phí, nó chỉ khai báo công cụ để chuyển text thành vector
EMBEDDING = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 3. KẾT NỐI VÀO DATABASE ĐÃ TỒN TẠI
# Tuyệt đối KHÔNG gọi hàm load_cv_data() hay quét PDF ở đây nữa
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