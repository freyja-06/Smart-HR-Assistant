"""
BaseLoader — Abstract Base Class cho tất cả Document Loaders.

Sử dụng Template Method Pattern:
  - get_docs() định nghĩa luồng chung: _load_raw() → _transform()
  - Subclass chỉ cần override _load_raw() và _transform()

Tuân thủ:
  - OCP: Thêm loader mới → tạo subclass mới, không sửa code cũ
  - DIP: Pipeline phụ thuộc vào BaseLoader (abstraction), không concrete class
  - LSP: Mọi subclass đều dùng được qua interface BaseLoader
"""

from abc import ABC, abstractmethod
from typing import List, Any
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class cho tất cả các loại document loader."""

    def get_docs(self, directory_path: str) -> List[Document]:
        """
        Template Method — định nghĩa luồng xử lý chung.

        1. _load_raw(): Load dữ liệu thô từ thư mục
        2. _transform(): Chuyển đổi raw data → LangChain Documents

        Subclass KHÔNG nên override method này.
        """


        logger.info(f"[{self.__class__.__name__}] Bắt đầu load từ: {directory_path}")

        raw_data = self._load_raw(directory_path)

        if not raw_data:
            logger.warning(f"[{self.__class__.__name__}] Không tìm thấy dữ liệu.")
            return []

        documents = self._transform(raw_data)

        logger.info(
            f"[{self.__class__.__name__}] Hoàn tất. "
            f"Tạo được {len(documents)} LangChain Documents."
        )


        return documents

    @abstractmethod
    def _load_raw(self, directory_path: str) -> Any:
        """
        Subclass triển khai: Load dữ liệu thô từ thư mục.

        Trả về dạng dữ liệu tùy ý (list, tuple, dict...)
        sẽ được truyền cho _transform().
        """
        ...

    @abstractmethod
    def _transform(self, raw_data: Any) -> List[Document]:
        """
        Subclass triển khai: Chuyển raw data thành List[LangChain Document].
        """
        ...
