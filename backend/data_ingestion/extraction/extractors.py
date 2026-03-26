"""
Tương tác với LLM để trích xuất dữ liệu từ CV text.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from backend.data_ingestion.schemas import CandidateProfile
from backend.agents.llm_processor.llm_factory import ModelFactory

load_dotenv()
logger = logging.getLogger(__name__)

# Cache the chain globally
_chain = None

def get_extractor_chain():
    """Khởi tạo LLM chain trễ (Lazy initialization). Singleton logic """
    global _chain
    if _chain is None:
        llm = ModelFactory.create(
            model_type="llm", 
            provider="ollama", 
            model_name="qwen2.5:3b"
        )
        llm_with_structed_output = llm.with_structured_output(CandidateProfile)

        instruction = """
            You are an expert HR Data Extractor.
            Extract information STRICTLY following the schema.

            IMPORTANT:
            - This is ONLY a PART of the CV (chunk)
            - Extract only information present in this chunk
            - Do NOT hallucinate missing fields

            Rules:
            - Return valid JSON only
            - Missing → None or empty list
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", instruction),
            ("human", "{cv_text}")
        ])

        _chain = prompt | llm_with_structed_output
    return _chain

def extract_chunk(chunk: str, retries: int = 2) -> CandidateProfile | None:
    """Gửi một đoạn text cho LLM để trích xuất dữ liệu."""
    chain = get_extractor_chain()
    for attempt in range(retries):
        try:
            return chain.invoke({"cv_text": chunk})
        except Exception as e:
            logger.error(f"Lỗi extract_chunk (Lần thử {attempt + 1}/{retries}): {e}")
            continue
    return None
