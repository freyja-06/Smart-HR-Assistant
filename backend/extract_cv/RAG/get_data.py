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

# 🔹 Khởi tạo embedding 1 lần
EMBEDDING = ModelFactory.create(
    model_type="embedding",
    provider="ollama",
    model_name="nomic-embed-text",
)

# 🔹 Path lưu vector DB
CHROMA_DIR = const.CHROMA_DIR

cv_docs = get_cv_Docs(CV_PATH)
company_docs = get_company_Docs(COMPANY_DOCS_PATH)

# Gán ID deterministic, tránh bị trùng embedding, vector db không bị phình to
def doc_id(text):
    return hashlib.md5(text.encode()).hexdigest()


def data_preparation(
    docs: list,
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

    # 🔥 TẠO EMBEDDING NGAY TẠI INDEX
    texts = [doc.page_content for doc in splits]
    doc_embeddings = EMBEDDING.embed_documents(texts)

    # ------------------------
    # Create or load Chroma
    # ------------------------
    if CHROMA_DIR.exists():
        print("load chroma + add docs")
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
        print("create chroma")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=EMBEDDING,
            persist_directory=str(CHROMA_DIR),
            collection_name=collection_name 
        )

    # 👉 return đủ để dùng sau này
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


