"""
LangChain Documents storage (.pkl)
"""

import os
import pickle
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


def save_langdocs(docs: List[Document], folder_path: str, var_name: str) -> None:
    """Lưu danh sách LangChain Documents xuống file .pkl"""
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{var_name}.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(docs, f)

    logger.info(f"Đã lưu {var_name} tại: {file_path}")


def load_langdocs(folder_path: str, var_name: str):
    """Load danh sách LangChain Documents từ file .pkl"""
    file_path = os.path.join(folder_path, f"{var_name}.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    logger.info(f"Đã load {var_name} từ: {file_path}")
    return data
