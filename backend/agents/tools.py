from typing import List, Callable
from langchain_core.tools import tool
from backend.config.database import cv_store, policies_store
from backend.extract_cv.RAG.rag_backend import get_retriever

@tool
def send_retriever(
    query: str,
    db_type: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    search_type: str = "mmr"
) -> str:
    """
    Retrieve documents from vector database.

    Args:
        query: search query
        db_type: "cv" or "policy"
        k: number of documents returned
        fetch_k: number of candidate documents
        lambda_mult: diversity parameter for MMR
        search_type: similarity | mmr
    """

    # chọn database
    if db_type.lower() == "cv":
        vectorstore = cv_store
    elif db_type.lower() == "policy":
        vectorstore = policies_store
    else:
        return "db_type phải là 'cv' hoặc 'policy'"

    try:
        retriever = get_retriever(
            vectorstore=vectorstore,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            search_type=search_type,
        )

        docs = retriever.invoke(query)

    except Exception as e:
        return f"Lỗi retriever: {str(e)}"

    if not docs:
        return "Không tìm thấy tài liệu"

    result = []

    for i, doc in enumerate(docs, 1):
        result.append(
            f"""
            --- Document {i} ---
            Content: {doc.page_content}
            Metadata: {doc.metadata}
            """
        )

    return f"Found {len(docs)} documents\n" + "\n".join(result)


tools: List[Callable] = [send_retriever]