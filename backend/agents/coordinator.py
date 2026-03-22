from langchain_core.prompts import ChatPromptTemplate
from backend.agents.llm_processor.llm_factory import LLMManager
from pydantic import BaseModel, Field
from typing import Literal, List, Any, Dict, Optional
from langchain_core.runnables import RunnableSequence, chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

llm = LLMManager.get_llm_with_fallbacks(
    temperature=0,
    max_tokens=2048,
    num_ctx=8192
)

class Task(BaseModel):
    task_id: int = Field(
        ge=0,
        description="ID duy nhất cho tác vụ. Bắt đầu từ 0."
    )
    route: Literal["RAG_SEARCH", "WRITE_EMAIL", "GENERATE_INTERVIEW_BRIEF"] = Field(
        description="""
        Phân loại yêu cầu của người dùng. 
        """
    )
    instruction: str = Field(
        description="Chỉ thị chi tiết cho tác vụ này. Nêu rõ đối tượng và công việc cần thực hiện."
    )
    args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Các tham số tùy chọn được trích xuất từ câu lệnh (VD: {'candidate_count': 10, 'skill': 'Python', 'is_batch': True})."
    )
    data_source: Literal["HISTORY_CV_DATABASE", "CV_DATABASE", "COMPANY_DOCS_DATABASE",  "INPUT", "UPSTREAM_TASK", "NONE"] = Field(
        description="""
        Xác định nguồn dữ liệu mà tác vụ này cần sử dụng:
        - 'UPSTREAM_TASK': Cần sử dụng kết quả đầu ra từ một tác vụ khác trong cùng một Plan này (phải khai báo ID vào dependencies).
        - 'HISTORY': Dữ liệu đã có sẵn từ vector database của các phiên chat/lệnh RAG trước đó (VD: 'Dựa vào 100 CV vừa tìm...'). Bỏ qua trường hợp này nếu history_cv_documents rỗng
        - 'RAG': Cần thực hiện truy vấn mới vào kho dữ liệu lớn để tìm kiếm thông tin chưa có sẵn.
        - 'INPUT': Dữ liệu do người dùng cung cấp trực tiếp trong lượt chat hiện tại (VD: người dùng upload 3 file CV và yêu cầu tóm tắt).
        - 'NONE': Không phụ thuộc vào dữ liệu ngoài (các câu hỏi giao tiếp chung chung).
        """
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="Danh sách task_id của các tác vụ phải hoàn thành trước khi tác vụ này chạy."
    )

class Plan(BaseModel):
    tasks: Dict[int, Task] = Field(
        description="""
    Danh sách (Dictionary mapping task_id → Task) các nhiệm vụ cần thực hiện, tạo thành một kế hoạch hoàn chỉnh.
    Nếu yêu cầu thiếu thông tin cần thiết, ví dụ như việc bảo gửi email nhưng không rõ gửi cho ai và nội dung gì
    hãy trả về danh sách rỗng!
        """
    )

manager_llm = llm.with_structured_output(Plan)

manager_instruction = """

You are the MANAGER agent of an HR AI system.

Your job is to analyze the user's request and create an execution plan
that other agents will follow.

You DO NOT answer the user.

You ONLY generate a structured plan in the required Pydantic format.

Available modules:

1. RAG_SEARCH
Use when the user needs information from:
- candidate CV database
- company HR policies
- stored HR documents

2. WRITE_EMAIL
Use when the user asks to:
- write an email
- reply to a candidate
- schedule interviews
- send HR communications

3. GENERATE_INTERVIEW_BRIEF
Use when the user asks to:
- prepare interview documents
- summarize candidate CV
- generate interview questions
- create hiring brief


Planning Rules:

1. Break complex requests into multiple tasks
2. Order tasks logically ()
3. Use RAG before generating summaries when external knowledge is needed

Each task must contain:
- task_id
- route
- instruction
- optional args

Return ONLY the structured Plan object, with tasks as a dictionary where:
- key = task_id (int, same as Task's object)
- value = Task object
- task_id must start from 0 and increase sequentially.

"""


manager_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{manager_instruction}"),
    ("human", "{user_input}")
])
manager_agent: RunnableSequence  = manager_prompt | manager_llm



MAX_DOCS = 5
MAX_DOC_CHARS = 1200

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into readable context for the LLM."""
    formatted = []

    for i, doc in enumerate(docs[:MAX_DOCS], start=1):
        content = doc.page_content[:MAX_DOC_CHARS]

        source = ""
        if doc.metadata:
            source = doc.metadata.get("source", "")
            if source:
                source = f"(source: {source})"

        formatted.append(
            f"Document {i} {source}\n"
            f"{content}"
        )

    return "\n\n".join(formatted)

@chain
def get_final_prompt(state: dict) -> str:
    user_input = state.get("user_input", "")

    # Lấy đúng tên Key đã định nghĩa trong GraphState
    cv_docs = state.get("cv_documents") or []
    company_docs = state.get("company_documents") or []
    all_raw_docs = cv_docs + company_docs
    
    # Lấy context đã được nén
    company_compressed_context = state.get("company_compressed_context")

    email_draft = state.get("email_draft")
    email_sent = state.get("email_sent")
    interview_brief = state.get("interview_brief")
    interview_pdf_path = state.get("interview_pdf_path")
    module_outputs = state.get("module_outputs") or {}
    completed_tasks = state.get("completed_tasks") or []
    failed_tasks = state.get("failed_tasks") or []

    context_blocks = []

    # =========================================================
    # 1. USER QUESTION & EXECUTION SUMMARY
    # =========================================================
    context_blocks.append(f"USER QUESTION:\n{user_input}")
    context_blocks.append(
        f"EXECUTION SUMMARY\nCompleted tasks: {completed_tasks}\nFailed tasks: {failed_tasks}"
    )

    # =========================================================
    # 2. KNOWLEDGE CONTEXT (Fix Logic)
    # =========================================================
    # Ưu tiên ngữ cảnh đã nén sạch sẽ, nếu lỗi không có nén thì lùi về dùng format_docs thô
    if cv_docs:
        cv_text = format_docs(cv_docs)
        context_blocks.append(f"CANDIDATE CVs (Raw Documents)\n{cv_text}")

    # 2.2 Xử lý riêng Company Documents (Ưu tiên nén, fallback thô)
    if company_compressed_context and company_compressed_context != "Không tìm thấy ngữ cảnh liên quan trong tài liệu.":
        context_blocks.append(f"COMPANY POLICY/DOCS (Compressed Context)\n{company_compressed_context}")
    elif company_docs:
        company_text = format_docs(company_docs)
        context_blocks.append(f"COMPANY POLICY/DOCS (Raw Documents)\n{company_text}")

    # =========================================================
    # 3. MODULE OUTPUTS
    # =========================================================
    if email_draft:
        context_blocks.append(f"EMAIL DRAFT GENERATED\n{email_draft}")

    if email_sent is True:
        context_blocks.append("EMAIL STATUS: The email was successfully sent.")

    if interview_brief:
        context_blocks.append(f"INTERVIEW BRIEF GENERATED\n{interview_brief[:2000]}")

    if interview_pdf_path:
        context_blocks.append(f"INTERVIEW PDF PATH: {interview_pdf_path}")

    if module_outputs:
        context_blocks.append(f"OTHER MODULE OUTPUTS\n{module_outputs}")

    # =========================================================
    # 4. FINAL INSTRUCTION
    # =========================================================
    context_blocks.append(
        """
        Using the information above, generate the final response to the user.

        Guidelines:
        - Answer the user question directly.
        - Use the provided knowledge when relevant.
        - If an email draft exists, inform the user.
        - If an interview brief exists, summarize the key points.
        - If tasks failed, politely inform the user.

        Write a clear, professional HR assistant response.
        """
    )

    final_prompt = "\n\n".join(context_blocks)
    return {"final_prompt": final_prompt}


response_instruction = """
You are the final RESPONSE agent of an HR AI assistant.

Your job is to generate the final answer for the user.

You will receive:

- the original user query
- retrieved documents
- filtered documents
- results generated by previous modules
- email drafts or interview documents if available

Rules:

1. Use retrieved knowledge when available.
2. Prefer filtered_documents over raw documents.
3. If interview documents were generated, summarize them clearly.
4. If an email was drafted or sent, inform the user.
5. Always produce a clear HR-friendly response.

If multiple tasks were executed, synthesize the results into a single coherent answer.

Never invent candidate data.

Be concise, clear, and professional.

"""


response_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{response_instruction}"),
    ("human", "{final_prompt}")
])
response_agent: RunnableSequence = get_final_prompt | response_prompt | llm | StrOutputParser()


