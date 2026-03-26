import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from backend.data_ingestion.schemas import CandidateProfile
from backend.agents.llm_processor.llm_factory import ModelFactory

load_dotenv()
logger = logging.getLogger(__name__)

# Khởi tạo biến toàn cục để lưu trữ cache cho chain
_chain = None

def get_extractor_chain():
    """Hàm khởi tạo trễ (Lazy initialization) cho LLM chain. Chỉ tạo chain ở lần gọi đầu tiên."""
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
                - Merge with other chunks later

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


def load_pdf(file_path: str) -> str:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])
    return full_text

def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def merge_profiles(results):
    final = {}

    for res in results:
        if not res:
            continue

        for key, value in res.dict().items():
            if not value:
                continue

            # nếu là list → merge
            if isinstance(value, list):
                if key not in final:
                    final[key] = []
                final[key].extend(value)

            # nếu là string → giữ cái đầu tiên
            else:
                if key not in final:
                    final[key] = value

    return CandidateProfile(**final)


def extract_chunk(chunk, retries=2):
    chain = get_extractor_chain()
    for attempt in range(retries):
        try:
            return chain.invoke({"cv_text": chunk})
        except Exception as e:
            logger.error(f"Lỗi extract_chunk (Lần thử {attempt + 1}/{retries}): {e}")
            continue
    return None


def process_single_cv(pdf_file):
    try:
        cv_text = load_pdf(pdf_file)

        # limit size để tránh overload
        cv_text = cv_text[:12000]

        chunks = chunk_text(cv_text)

        results = []
        for chunk in chunks:
            res = extract_chunk(chunk)
            if res:
                results.append(res)

        return merge_profiles(results)

    except Exception as e:
        print(f"Lỗi file {pdf_file}: {e}")
        return None


# Có thể có rủi ro sập tiến trình nạp CV hàng loạt
