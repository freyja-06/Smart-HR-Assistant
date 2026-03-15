from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from extract_cv.RAG.convert_pydantic_to_langDocs import get_cv_Docs, get_policies_Docs
import hashlib
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# 🔹 Khởi tạo embedding 1 lần
EMBEDDING = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 🔹 Path lưu vector DB
CHROMA_DIR = BASE_DIR / "src" / "chroma_db"

# Gán ID deterministic, tránh bị trùng embedding, vector db không bị phình to
def doc_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def data_preparation(
    docs: list[Document],
    collection_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = splitter.split_documents(docs)
    ids = [doc_id(doc.page_content) for doc in splits]

    # ------------------------
    # Create or load Chroma
    # ------------------------
    if CHROMA_DIR.exists():
        # Thêm collection_name vào đây
        print("tạo chroma!")
        vectorstore = Chroma(
            collection_name=collection_name, 
            persist_directory=str(CHROMA_DIR),
            embedding_function=EMBEDDING
        )
        vectorstore.add_documents(
            documents=splits, 
            ids=ids
        )

        return vectorstore
    else:
        # Thêm collection_name vào đây
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=EMBEDDING,
            persist_directory=str(CHROMA_DIR),
            collection_name=collection_name 
        )
        return vectorstore



def load_cv_data(
    directory_path: str = BASE_DIR / "data" / "All_pdf_cv",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    print("Đường dẫn: " , end = "")
    print(directory_path)

    docs = get_cv_Docs(directory_path)
    # Truyền tên Collection là "CVs"
    return data_preparation(docs, collection_name="CVs", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_policies_data(
    directory_path: str = BASE_DIR / "data" / "Client_policies",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    docs = get_policies_Docs(directory_path)
    # Truyền tên Collection là "Policies"
    return data_preparation(docs, collection_name="Policies", chunk_size=chunk_size, chunk_overlap=chunk_overlap)



def get_retriever(
    k: int = 6,
    fetch_k: int = 50,
    lambda_mult: float = 0.4,
    *,
    vectorstore,
    search_type: str = "mmr",
) -> VectorStoreRetriever:
    # ------------------------
    # Validate params
    # ------------------------
    if fetch_k < k:
        raise ValueError("fetch_k must be >= k")

    if not (0.0 <= lambda_mult <= 1.0):
        raise ValueError("lambda_mult must be between 0 and 1")

    # ------------------------
    # Retriever
    # ------------------------
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )

    return retriever