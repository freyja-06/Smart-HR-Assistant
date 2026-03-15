from dotenv import load_dotenv
from backend.agents.llm_processor.llm_factory import LLMManager

load_dotenv()

llm = LLMManager.get_llm_with_fallbacks(
    temperature=0,
    max_output_tokens=2048
)