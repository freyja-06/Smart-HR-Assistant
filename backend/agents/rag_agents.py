from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from backend.agents.llm_processor.llm import llm
from pydantic import BaseModel, Field
from typing import Literal, List

optimize_query_instruction = """
You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
"""


optimize_query_prompt = ChatPromptTemplate.from_messages([
    ("system", optimize_query_instruction),
    ("human", "{user_query}")
])


optimize_query_agent = optimize_query_prompt | llm | StrOutputParser()

class SubQuery(BaseModel):
    sub_query: str = Field(
        description="A specific, standalone question derived from the original query"
    )
    data_source: Literal[
        "HISTORY_CV_DATABASE",
        "CV_DATABASE",
        "COMPANY_DOCS_DATABASE"
    ] = Field(
        description="""
        Choose the most appropriate data source:
        - HISTORY_CV_DATABASE: past user interactions, previous CV searches
        - CV_DATABASE: candidate CV/resume data
        - COMPANY_DOCS_DATABASE: internal company documents, policies, job descriptions
        """
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="""
        Fusion weight between semantic search and keyword search.

        - 0.0 → rely entirely on keyword-based retrieval (BM25)
        - 1.0 → rely entirely on semantic embedding retrieval
        - 0.5 → balanced hybrid search

        Use higher alpha for conceptual or vague queries.
        Use lower alpha for exact keyword matching queries.
        """
    )

    k: int = Field(
        default=None,
        ge=1,
        le=100,
        description="""
        Number of top documents to retrieve from the database. 

        - Small k (e.g., 5-10): higher precision, faster
        - Large k (e.g., 20-50): higher recall, useful for reranking

        Increase k when the query is broad or ambiguous.
        k should typically be 2 - 3x larger than the final number of desired results
        to ensure sufficient recall before reranking.

        ONLY provide this number if the user EXPLICITLY asks for a specific quantity (e.g., "find me 5 CVs", "top 3 candidates").
        If the user does NOT specify a quantity, DO NOT provide this value (leave it null/None).

        """
    )


class ListSubQuery(BaseModel):
    queries: List[SubQuery] = Field(
        description="List of 1-4 sub queries"
    )

sub_query_instruction = """
You are an AI assistant that decomposes a query AND routes each sub-query to the correct data source.

Tasks:
1. Break the query into 1-4 sub-queries
2. Each sub-query must be:
   - Clear
   - Searchable
   - Independent
3. Assign the most relevant data_source for each sub-query

Rules:
- Prefer CV_DATABASE when querying candidate info
- Prefer COMPANY_DOCS_DATABASE for job descriptions, requirements
- Prefer HISTORY_CV_DATABASE if related to past searches or user history

Return ONLY valid JSON.
"""


sub_query_prompt = ChatPromptTemplate.from_messages([
    ("system", sub_query_instruction),
    ("human", "{query}")
])
structured_llm = llm.with_structured_output(ListSubQuery)
# Tối ưu lại việc ép kiểu cho llm

sub_query_agent: RunnableSequence = (
    optimize_query_agent
    | (lambda optimized_query: {"query": optimized_query})
    | sub_query_prompt
    | structured_llm
)


context_compressor_instruction = """

    Bạn là một Context Compressor trong hệ thống hỏi đáp tài liệu nội bộ.

    Mục tiêu:
    Tối ưu hóa context để giúp LLM trả lời chính xác nhất.

    Nguyên tắc:
    1. CHỈ giữ thông tin liên quan trực tiếp đến câu hỏi.
    2. LOẠI BỎ:
    - thông tin dư thừa
    - ví dụ không cần thiết
    - phần giải thích dài dòng
    3. GIỮ:
    - dữ kiện quan trọng
    - số liệu, tên riêng, quy trình, logic
    4. KHÔNG:
    - suy diễn
    - thêm kiến thức ngoài
    - thay đổi ý nghĩa
    5. Nếu tài liệu KHÔNG liên quan → trả về: "IRRELEVANT"

    Định dạng output:
    - Bullet points (ưu tiên)
    - Ngắn gọn, rõ ràng
"""
