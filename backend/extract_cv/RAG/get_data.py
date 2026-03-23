from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from backend.extract_cv.RAG.convert_pydantic_to_langDocs import get_cv_Docs, get_company_Docs
import hashlib
from dotenv import load_dotenv
from backend.extract_cv.RAG.bm25_module import create_bm25_index, save_bm25
import backend.constant_variables as const
from backend.agents.llm_processor.llm_factory import ModelFactory

load_dotenv()

CV_PATH: str = const.CV_PATH
COMPANY_DOCS_PATH: str = const.COMPANY_DOCS_PATH
BM25_CV_PATH = const.BM25_CV_PATH
BM25_COMPANY_PATH = const.BM25_COMPANY_PATH

print("[DEBUG] GET_DATA: Khởi tạo Model Embedding...")
# 🔹 Khởi tạo embedding 1 lần
EMBEDDING = ModelFactory.create(
    model_type="embedding",
    provider="ollama",
    model_name="nomic-embed-text",
)
print("[DEBUG] GET_DATA: Khởi tạo Model Embedding XONG!")


# 🔹 Path lưu vector DB
CHROMA_DIR = const.CHROMA_DIR

print("[DEBUG] GET_DATA: Đang lấy cv_docs ở global level...")
cv_docs = get_cv_Docs(CV_PATH)
print(f"[DEBUG] GET_DATA: Lấy cv_docs xong! Tổng cộng: {len(cv_docs)} tài liệu.")

print("[DEBUG] GET_DATA: Đang lấy company_docs ở global level...")
company_docs = get_company_Docs(COMPANY_DOCS_PATH)
print(f"[DEBUG] GET_DATA: Lấy company_docs xong! Tổng cộng: {len(company_docs)} tài liệu.")

# Gán ID deterministic, tránh bị trùng embedding, vector db không bị phình to
def doc_id(text):
    return hashlib.md5(text.encode()).hexdigest()


def data_preparation(
    docs: list,
    collection_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    
    if not docs:
        print(f"[DEBUG] data_preparation: Không có dữ liệu cho {collection_name}. Bỏ qua.")
        return None, []

    print(f"\n[DEBUG] data_preparation: Bắt đầu chia nhỏ (split) cho {collection_name}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    splits = splitter.split_documents(docs)
    ids = [doc_id(doc.page_content) for doc in splits]
    print(f"[DEBUG] data_preparation: Đã cắt thành {len(splits)} chunks. Bắt đầu gọi Ollama Embedding (Bước này có thể mất nhiều thời gian)...")


    # 🔥 TẠO EMBEDDING NGAY TẠI INDEX
    texts = [doc.page_content for doc in splits]
    doc_embeddings = EMBEDDING.embed_documents(texts)
    print("[DEBUG] data_preparation: Embedding XONG! Bắt đầu lưu vào ChromaDB...")

    # ------------------------
    # Create or load Chroma
    # ------------------------
    if CHROMA_DIR.exists():
        print(f"[DEBUG] data_preparation: Đang Load Chroma và thêm {len(splits)} tài liệu vào collection '{collection_name}'...")
        vectorstore = Chroma(
            collection_name=collection_name, 
            persist_directory=str(CHROMA_DIR),
            embedding_function=EMBEDDING
        )
        vectorstore.add_documents(
            documents=splits, 
            ids=ids
        )
    else:
        print(f"[DEBUG] data_preparation: Đang TẠO MỚI Chroma collection '{collection_name}'...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=EMBEDDING,
            persist_directory=str(CHROMA_DIR),
            collection_name=collection_name 
        )

    print("[DEBUG] data_preparation: Lưu ChromaDB HOÀN TẤT.")
    return vectorstore, doc_embeddings


def load_cv_data(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    print("Đường dẫn: ", end="")
    print(CV_PATH)

    docs = cv_docs

    vectorstore, doc_embeddings = data_preparation(
        docs,
        collection_name="CVs",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return vectorstore, doc_embeddings, docs


def load_company_docs_data(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    docs = company_docs

    vectorstore, doc_embeddings = data_preparation(
        docs,
        collection_name="company_docs",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return vectorstore, doc_embeddings, docs


def load_cv_bm25_index():
    docs = cv_docs
    bm25_data = create_bm25_index(docs)
    save_bm25(bm25_data["bm25"], bm25_data["corpus"], BM25_CV_PATH)
    

def load_company_docs_bm25_index():
    docs = company_docs
    bm25_data = create_bm25_index(docs)
    save_bm25(bm25_data["bm25"], bm25_data["corpus"], BM25_COMPANY_PATH)


