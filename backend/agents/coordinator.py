from langchain_core.prompts import ChatPromptTemplate
from backend.agents.llm_processor.llm_factory import ModelFactory
from pydantic import BaseModel, Field
from typing import Literal, List, Any, Dict, Optional
from langchain_core.runnables import RunnableSequence, chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

llm = ModelFactory.create(
        model_type="llm", 
        provider="ollama", 
        model_name="qwen2.5", 
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
    tasks: List[Task] = Field(
        default_factory=list,
        description="""
    Danh sách các nhiệm vụ cần thực hiện, tạo thành một kế hoạch hoàn chỉnh.
    Nếu yêu cầu thiếu thông tin cần thiết, ví dụ như việc bảo gửi email nhưng không rõ gửi cho ai và nội dung gì
    hãy trả về danh sách rỗng!
        """
    )

manager_llm = llm.with_structured_output(Plan)

manager_instruction = """
Bạn là agent QUẢN LÝ (Manager) của hệ thống HR AI.

Nhiệm vụ: Phân tích yêu cầu của người dùng và tạo kế hoạch thực thi (Plan)
cho các agent khác thực hiện.

Bạn KHÔNG trả lời trực tiếp cho người dùng.
Bạn CHỈ tạo ra một Plan có cấu trúc theo đúng Pydantic schema.

Các module có sẵn:

1. RAG_SEARCH
   Sử dụng khi người dùng cần tìm kiếm thông tin từ:
   - Cơ sở dữ liệu CV ứng viên
   - Chính sách HR, quy trình công ty
   - Tài liệu HR nội bộ

2. WRITE_EMAIL
   Sử dụng khi người dùng yêu cầu:
   - Viết email
   - Trả lời ứng viên
   - Lên lịch phỏng vấn
   - Gửi thông báo HR

3. GENERATE_INTERVIEW_BRIEF
   Sử dụng khi người dùng yêu cầu:
   - Chuẩn bị tài liệu phỏng vấn
   - Tóm tắt CV ứng viên
   - Tạo câu hỏi phỏng vấn
   - Tạo hồ sơ tuyển dụng

Quy tắc lập kế hoạch:

1. Chia yêu cầu phức tạp thành nhiều task nhỏ.
2. Sắp xếp task theo thứ tự logic (task phụ thuộc phải chạy sau).
3. Dùng RAG_SEARCH trước khi tạo tóm tắt nếu cần dữ liệu từ database.
4. QUAN TRỌNG: Trường 'instruction' của mỗi task PHẢI viết bằng tiếng Việt.
   TUYỆT ĐỐI KHÔNG viết instruction bằng tiếng Trung (Chinese) hay bất kỳ ngôn ngữ nào khác ngoài tiếng Việt.

Mỗi task phải chứa:
- task_id (bắt đầu từ 0, tăng dần)
- route
- instruction (bằng tiếng Việt)
- args (tùy chọn)
- data_source
- dependencies

Trả về DUY NHẤT một Plan object hợp lệ.
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
        Dựa trên thông tin ở trên, hãy tạo câu trả lời cuối cùng cho người dùng.

        Hướng dẫn:
        - Trả lời BẰNG TIẾNG VIỆT.
        - Trả lời trực tiếp câu hỏi của người dùng.
        - Sử dụng dữ liệu truy xuất được khi có liên quan.
        - Nếu có bản nháp email, hãy thông báo cho người dùng.
        - Nếu có tài liệu phỏng vấn, tóm tắt các điểm chính.
        - Nếu có task thất bại, thông báo lịch sự cho người dùng.
        - KHÔNG viết bằng tiếng Trung (Chinese).

        Viết câu trả lời rõ ràng, chuyên nghiệp, phù hợp với vai trò trợ lý HR.
        """
    )

    final_prompt = "\n\n".join(context_blocks)
    return {"final_prompt": final_prompt}


response_instruction = """
Bạn là agent TRẢ LỜI CUỐI CÙNG (Response Agent) của hệ thống trợ lý HR AI.

Nhiệm vụ: Tạo câu trả lời cuối cùng cho người dùng dựa trên dữ liệu đã thu thập.

Bạn sẽ nhận được:
- Câu hỏi gốc của người dùng
- Tài liệu đã truy xuất (retrieved documents)
- Kết quả từ các module trước đó
- Bản nháp email hoặc tài liệu phỏng vấn (nếu có)

Quy tắc:
1. Sử dụng dữ liệu truy xuất được khi có sẵn.
2. Nếu có tài liệu phỏng vấn, tóm tắt rõ ràng.
3. Nếu email đã được soạn hoặc gửi, thông báo cho người dùng.
4. TUYỆT ĐỐI KHÔNG bịa đặt dữ liệu ứng viên.
5. Nếu nhiều task đã chạy, tổng hợp kết quả thành MỘT câu trả lời mạch lạc.
6. LUÔN trả lời bằng TIẾNG VIỆT.
7. KHÔNG viết bằng tiếng Trung (Chinese) dưới bất kỳ hình thức nào.

Phong cách: Ngắn gọn, rõ ràng, chuyên nghiệp, phù hợp với vai trò HR.
"""


response_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{response_instruction}"),
    ("human", "{final_prompt}")
])
response_agent: RunnableSequence = get_final_prompt | response_prompt | llm | StrOutputParser()

