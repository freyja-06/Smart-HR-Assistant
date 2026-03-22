from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from backend.agents.llm_processor.llm_factory import LLMManager # Thay đổi import
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import asyncio

llm = LLMManager.get_llm_with_fallbacks(
    temperature=0,
    max_tokens=2048,
    num_ctx=8192
)

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

    k: Optional[int] = Field(
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
structured_llm = LLMManager.get_llm_with_fallbacks(
    pydantic_schema=ListSubQuery,
    temperature=0,
    max_tokens=2048,
    num_ctx=8192
)
# Tối ưu lại việc ép kiểu cho llm

sub_query_agent: RunnableSequence = (
     (lambda raw_input: {"user_query": raw_input})
    | optimize_query_agent
    | (lambda optimized_query: {"query": optimized_query})
    | sub_query_prompt
    | structured_llm
)

map_instruction = """
Bạn là một chuyên gia phân tích dữ liệu. Nhiệm vụ của bạn là trích xuất thông tin từ tài liệu công ty được cung cấp để trả lời câu hỏi của người dùng.
Trước hết tôi nhắc một điều quan trọng: Chúng ta chỉ đang xét các truy vấn liên quan đến tài liệu của công ty. 
Nếu trong mục 'Câu hỏi' có các lệnh truy vấn liên quan tới viêc lọc cv, hãy bỏ qua tất cả chúng!

Nguyên tắc Map:
1. Đọc kỹ tài liệu và đối chiếu với câu hỏi. 
2. CHỈ trích xuất các dữ kiện, số liệu, quy trình có liên quan TRỰC TIẾP đến câu hỏi.
3. Giữ nguyên các thuật ngữ chuyên ngành và tên riêng.
4. KHÔNG suy diễn hoặc thêm thông tin bên ngoài.
5. Định dạng đầu ra: Bullet points ngắn gọn.
6. Quan trọng: Nếu tài liệu hoàn toàn KHÔNG chứa thông tin liên quan đến câu hỏi, hãy trả về CHÍNH XÁC một từ: "IRRELEVANT".
"""

map_prompt = ChatPromptTemplate.from_messages([
    ("system", map_instruction),
    ("human", "Câu hỏi: {query}\n\nTài liệu:\n{raw_documents}")
])


reduce_instruction = """
Bạn là một biên tập viên tổng hợp ngữ cảnh (Context Compressor). Bạn sẽ nhận được các đoạn thông tin đã được trích xuất từ nhiều nguồn tài liệu khác nhau.

Trước hết tôi nhắc một điều quan trọng: Chúng ta chỉ đang xét các truy vấn liên quan đến tài liệu của công ty. 
Nếu trong mục 'Câu hỏi gốc' có các lệnh truy vấn liên quan tới viêc lọc cv, hãy bỏ qua tất cả chúng!


Nhiệm vụ của bạn:
1. Tổng hợp tất cả các đoạn thông tin này thành một ngữ cảnh duy nhất, thống nhất để trả lời câu hỏi.
2. LOẠI BỎ hoàn toàn các thông tin trùng lặp giữa các nguồn.
3. Tổ chức lại thông tin theo cấu trúc logic (ví dụ: gộp các ý giống nhau, sắp xếp theo trình tự nếu có).
4. Định dạng đầu ra cuối cùng: Bullet points, cực kỳ cô đọng, rõ ràng, không dài dòng.
5. KHÔNG tự trả lời câu hỏi, chỉ đóng vai trò nén và làm sạch ngữ cảnh.
"""

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", reduce_instruction),
    ("human", "Câu hỏi gốc: {query}\n\nCác đoạn thông tin đã trích xuất:\n{mapped_contexts}")
])

async def async_map_compress(query: str, docs_chunk: List, llm) -> str:
    """Xử lý nén một cụm tài liệu (Map)"""

    # Gộp nội dung các docs trong chunk thành một string
    raw_text = "\n\n---\n\n".join([doc.page_content for doc in docs_chunk])
    

    chain = map_prompt | llm | StrOutputParser()
    
    # Sử dụng ainvoke để chạy bất đồng bộ
    result = await chain.ainvoke({
        "query": query,
        "raw_documents": raw_text
    })
    
    return result.strip()

async def parallel_map_reduce_compress(
    query: str, 
    docs: List, 
    llm, 
    batch_size: int = 5
) -> str:
    """
    Hàm chính thực thi Map-Reduce
    batch_size: số lượng tài liệu trong mỗi cụm (khuyến nghị 5-6 để tối ưu cho model 7B)
    """
    if not docs:
        return ""

    # Bước 1: Chia nhỏ 30 tài liệu thành các cụm (chunks)
    doc_chunks = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    
    # Bước 2: MAP - Tạo các task chạy song song cho từng cụm
    map_tasks = [async_map_compress(query, chunk, llm) for chunk in doc_chunks]
    
    # Đợi tất cả các task hoàn thành
    mapped_results = await asyncio.gather(*map_tasks)
    
    # Bước 3: Lọc bỏ các kết quả không liên quan
    valid_contexts = [
        res for res in mapped_results 
        if res.strip().upper() != "IRRELEVANT" and len(res.strip()) > 0
    ]
    
    # Nếu không có tài liệu nào liên quan, trả về chuỗi rỗng
    if not valid_contexts:
        return "Không tìm thấy ngữ cảnh liên quan trong tài liệu."

    # Bước 4: REDUCE - Tổng hợp lại các context hợp lệ
    merged_contexts_str = "\n\n".join(
        [f"Nguồn {i+1}:\n{ctx}" for i, ctx in enumerate(valid_contexts)]
    )
    
    reduce_chain = reduce_prompt | llm | StrOutputParser()
    
    final_compressed_context = await reduce_chain.ainvoke({
        "query": query,
        "mapped_contexts": merged_contexts_str
    })
    
    return final_compressed_context

async def context_compressor_agent(query, company_docs):
    
    if not company_docs:
        return ""

    # Chạy hàm Map-Reduce mà ta đã viết trước đó
    compressed_ctx = await parallel_map_reduce_compress(
        query=query, 
        docs=company_docs, 
        llm=llm
    )
    
    return compressed_ctx