from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from agents.tools import tools
from dotenv import load_dotenv

load_dotenv()

class LLMFactory:
    """Factory để khởi tạo các model LLM khác nhau."""

    @staticmethod
    def create_model(provider: str, model_name: str, **kwargs):
        provider = provider.lower()

        if model_name == "gemini-2.5-flash" or model_name == "gemini-1.5-pro":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature = kwargs.get("temperature", 0),
                max_tokens = kwargs.get("max_tokens", 2048),
                timeout = kwargs.get("timeout", None),
                max_retries = kwargs.get("max_retries", 2)
                # other params...
            )

        elif model_name == "qwen2.5":
            return ChatOllama(
                model=model_name,
                temperature = kwargs.get("temperature", 0),
                num_predict = kwargs.get("max_tokens", 2048),
                timeout = kwargs.get("timeout", None),
                max_retries = kwargs.get("max_retries", 2)
            )
        else:
            raise ValueError(f"Provider '{provider}' chưa được hỗ trợ.")


class LLMManager:
    """Quản lý LLM và cấu hình Fallback."""
    
    @staticmethod
    def get_llm_with_fallbacks(**kwargs):
        """
        Khởi tạo model chính và cấu hình các model dự phòng.
        Nếu model chính lỗi, nó sẽ tự động chạy model dự phòng theo thứ tự.
        """
        primary_llm = LLMFactory.create_model(
            provider="ollama", 
            model_name="qwen2.5",
            **kwargs
        )

        fallback_1 = LLMFactory.create_model(
            provider="google", 
            model_name="gemini-1.5-pro",
            **kwargs 
        )

        fallback_2 = LLMFactory.create_model(
            provider="google", 
            model_name="gemini-2.5-flash",
            **kwargs
        )

        llm_with_fallback = primary_llm.with_fallbacks([fallback_1, fallback_2])

        return llm_with_fallback
