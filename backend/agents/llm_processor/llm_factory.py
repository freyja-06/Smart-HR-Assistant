from typing import Any, Dict, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from backend.agents.llm_processor.ollama_cross_encoder import OllamaCrossEncoder
import json


class BaseModelBuilder:
    def build(self, provider: str, model_name: str, **kwargs) -> Any:
        raise NotImplementedError()


class LLMBuilder(BaseModelBuilder):
    def build(self, provider: str, model_name: str, **kwargs) -> Any:
        provider = provider.lower()

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 2048),
                timeout=kwargs.get("timeout", None),
                max_retries=kwargs.get("max_retries", 2)
            )

        elif provider == "ollama":
            return ChatOllama(
                model=model_name,
                temperature=kwargs.get("temperature", 0),
                num_predict=kwargs.get("max_tokens", 2048),
                num_ctx=kwargs.get("num_ctx", 8192), # Thêm thông số num_ctx để mở rộng context window
                timeout=kwargs.get("timeout", None),
                max_retries=kwargs.get("max_retries", 2),
                keep_alive="3m"
            )

        raise ValueError(f"LLM Provider '{provider}' chưa được hỗ trợ.")


class EmbeddingBuilder(BaseModelBuilder):
    def build(self, provider: str, model_name: str, **kwargs) -> Any:
        provider = provider.lower()

        if provider == "google":
            return GoogleGenerativeAIEmbeddings(model=model_name)

        elif provider == "ollama":
            return OllamaEmbeddings(model=model_name)

        raise ValueError(f"Embedding Provider '{provider}' chưa được hỗ trợ.")


class CrossEncoderBuilder(BaseModelBuilder):
    def build(self, provider: str, model_name: str, **kwargs):

        if provider == "ollama":
            llm = ModelFactory.create(
                model_type="llm",
                provider="ollama",
                model_name=model_name,
                temperature=0,
                max_tokens=10
            )
            return OllamaCrossEncoder(llm)

        raise ValueError(f"Unsupported provider: {provider}")


class ModelFactory:
    _builders: Dict[str, BaseModelBuilder] = {
        "llm": LLMBuilder(),
        "embedding": EmbeddingBuilder(),
        "cross_encoder": CrossEncoderBuilder(),
    }

    # Cache singleton (class attribute)
    _instances: Dict[Tuple, Any] = {}

    @classmethod
    def _make_key(cls, model_type: str, provider: str, model_name: str, **kwargs):
        # Chuyển đổi kwargs thành chuỗi JSON để đảm bảo có thể băm (hash) an toàn
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        return (
            model_type.lower(),
            provider.lower(),
            model_name,
            kwargs_str
        )

    @classmethod
    def register_builder(cls, model_type: str, builder: BaseModelBuilder) -> None:
        cls._builders[model_type.lower()] = builder

    @classmethod
    def create(cls, model_type: str, provider: str, model_name: str, **kwargs) -> Any:
        """
        Factory method dùng để khởi tạo hoặc tái sử dụng (singleton) các model.

        Cơ chế hoạt động:
        -----------------
        - Mỗi model được xác định duy nhất bởi bộ tham số:
            (model_type, provider, model_name, kwargs)

        - Hàm sẽ:
            1. Tạo một key duy nhất từ các tham số trên
            2. Kiểm tra xem model với key này đã được khởi tạo chưa
                - Nếu RỒI → trả về instance đã tồn tại (Singleton)
                - Nếu CHƯA → tạo mới thông qua builder tương ứng
            3. Lưu instance vào cache (_instances)
            4. Trả về instance

        Singleton Behavior:
        -------------------
        - Đảm bảo mỗi cấu hình model chỉ được khởi tạo DUY NHẤT một lần
        - Các lần gọi sau với cùng cấu hình sẽ reuse lại instance
        - Giúp:
            + Tránh load lại model nặng (LLM, embedding)
            + Tăng performance
            + Giảm tài nguyên (RAM / GPU)

        Parameters:
        -----------
        model_type : str
            Loại model cần khởi tạo ("llm", "embedding", "cross_encoder")

        provider : str
            Nhà cung cấp model ("ollama", "google", ...)

        model_name : str
            Tên cụ thể của model (vd: "llama3", "gemini-pro", ...)

        **kwargs :
            Các tham số cấu hình bổ sung (temperature, max_tokens, ...)

        Returns:
        --------
        Any
            Instance của model đã được khởi tạo hoặc tái sử dụng

        Raises:
        -------
        ValueError:
            Nếu model_type chưa được đăng ký trong factory

        Gợi ý một số model hợp lý để test
            1. Normal LLM: 
                qwen2.5

            2. Embedding model
                nomic-embed-text

            3. Cross-encoder (reranker)
                sam860/qwen3-reranker:0.6b-Q8_0
        
    """
        key = cls._make_key(model_type, provider, model_name, **kwargs)

        # Nếu đã có → return luôn (Singleton)
        if key in cls._instances:
            return cls._instances[key]

        builder = cls._builders.get(model_type.lower())
        if not builder:
            raise ValueError(f"Loại model '{model_type}' chưa được đăng ký.")

        instance = builder.build(provider, model_name, **kwargs)

        # Lưu lại instance
        cls._instances[key] = instance

        return instance

class LLMManager:
    """Quản lý LLM và cấu hình Fallback."""
    
    @staticmethod
    def get_llm_with_fallbacks(pydantic_schema = None, **kwargs):

        """
        Chú ý: Không hỗ trợ Cross Encoder LLm
        """

        if "temperature" not in kwargs:
            kwargs["temperature"] = 0
        if "num_ctx" not in kwargs:
            kwargs["num_ctx"] = 8192

        primary_llm = ModelFactory.create(
                model_type="llm", 
                provider="ollama", 
                model_name="qwen2.5", 
                **kwargs
            )

        fallback_1 = ModelFactory.create(
            model_type="llm", 
            provider="google", 
            model_name="gemini-1.5-pro", 
            **kwargs
        )

        fallback_2 = ModelFactory.create(
            model_type="llm", 
            provider="google", 
            model_name="gemini-2.5-flash", 
            **kwargs
        )

        if pydantic_schema is not None:
            primary_llm = primary_llm.with_structured_output(pydantic_schema)
            fallback_1 = fallback_1.with_structured_output(pydantic_schema)
            fallback_2 = fallback_2.with_structured_output(pydantic_schema)

        return primary_llm.with_fallbacks([fallback_1, fallback_2])
    