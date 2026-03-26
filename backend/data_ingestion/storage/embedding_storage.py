"""
Embedding storage (.npy)
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_embeddings(embeddings, file_path: str) -> None:
    """Lưu embeddings xuống file .npy"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, embeddings)
    logger.info(f"Đã lưu embeddings tại: {file_path}")


def load_embeddings(file_path: str) -> np.ndarray:
    """Load embeddings từ file .npy"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    logger.info(f"Đã load embeddings từ: {file_path}")
    return data
