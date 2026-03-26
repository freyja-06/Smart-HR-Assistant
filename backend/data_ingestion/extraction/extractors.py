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
Bạn là một chuyên gia trích xuất dữ liệu HR (Human Resources Data Extractor).

NHIỆM VỤ: Phân tích đoạn văn bản CV bên dưới và trích xuất thông tin ứng viên theo đúng schema được quy định.

══════════════════════════════════════════
QUY TẮC QUAN TRỌNG:
══════════════════════════════════════════
1. Đây CHỈ LÀ MỘT PHẦN (chunk) của CV, KHÔNG phải toàn bộ CV.
2. CHỈ trích xuất thông tin CÓ MẶT trong đoạn text này.
3. TUYỆT ĐỐI KHÔNG bịa đặt, suy diễn, hoặc thêm thông tin không tồn tại.
4. Nếu một trường không có dữ liệu trong đoạn text → trả về None (cho string) hoặc [] (cho list).
5. Giữ nguyên ngôn ngữ gốc của nội dung CV (tiếng Việt giữ tiếng Việt, tiếng Anh giữ tiếng Anh).

══════════════════════════════════════════
HƯỚNG DẪN CHI TIẾT TỪNG TRƯỜNG:
══════════════════════════════════════════

■ full_name (str | None):
  - Họ và tên đầy đủ của ứng viên.
  - Thường xuất hiện ở đầu CV, in đậm hoặc cỡ chữ lớn.
  - Ví dụ: "Nguyễn Văn An", "Trần Thị Bình", "John Doe"
  - KHÔNG nhầm với tên công ty, tên trường học, hoặc tên người tham chiếu.

■ email (str | None):
  - Địa chỉ email liên hệ của ứng viên.
  - Ví dụ: "nguyenvanan@gmail.com"
  - KHÔNG lấy email của công ty/nhà tuyển dụng nếu có.

■ phone (str | None):
  - Số điện thoại liên hệ. Giữ nguyên định dạng gốc.
  - Ví dụ: "0901234567", "+84 912 345 678", "(028) 3456 7890"

■ summary (str | None):
  - Tóm tắt nghề nghiệp / mục tiêu nghề nghiệp / giới thiệu bản thân.
  - Thường nằm ở phần "Mục tiêu", "Objective", "Summary", "Giới thiệu", "About me".
  - Trích xuất NGUYÊN VĂN, không tóm tắt lại.

■ skills (List[str]):
  - Danh sách kỹ năng, công nghệ, ngôn ngữ lập trình, công cụ.
  - Mỗi phần tử là MỘT kỹ năng riêng biệt.
  - Ví dụ: ["Python", "React", "Quản lý dự án", "Microsoft Excel", "Tiếng Anh IELTS 7.0"]
  - KHÔNG gộp nhiều kỹ năng vào một chuỗi.

■ experiences (List[str]):
  - Mỗi phần tử mô tả MỘT kinh nghiệm làm việc.
  - Định dạng mong muốn: "Vị trí tại Công ty (thời gian): mô tả công việc"
  - Ví dụ: ["Software Engineer tại FPT Software (01/2022 - 12/2023): Phát triển hệ thống backend bằng Java Spring Boot, quản lý team 5 người"]
  - Nếu CV liệt kê chi tiết mô tả công việc, HÃY GỘP vào cùng một chuỗi với vị trí đó.
  - KHÔNG tách riêng mô tả công việc thành phần tử riêng biệt.

■ education (List[str]):
  - Mỗi phần tử mô tả MỘT bằng cấp / trình độ học vấn.
  - Định dạng mong muốn: "Bằng cấp - Chuyên ngành - Trường"
  - Ví dụ: ["Cử nhân - Khoa học Máy tính - Đại học Bách Khoa Hà Nội", "Thạc sĩ - AI - Stanford University"]
  - Nếu có GPA hoặc thời gian, kèm theo: "Cử nhân - CNTT - ĐH Công Nghệ (2018-2022, GPA: 3.5)"

══════════════════════════════════════════
NHỮNG ĐIỀU KHÔNG ĐƯỢC LÀM:
══════════════════════════════════════════
✗ KHÔNG bịa tên, email, số điện thoại nếu không thấy trong text.
✗ KHÔNG dịch nội dung từ ngôn ngữ này sang ngôn ngữ khác.
✗ KHÔNG trả về chuỗi rỗng "" — dùng None cho các trường optional.
✗ KHÔNG thêm kỹ năng mà ứng viên không liệt kê rõ ràng.
✗ KHÔNG viết bằng tiếng Trung (Chinese) dưới bất kỳ hình thức nào.

Hãy phân tích đoạn text CV sau và trả về kết quả JSON theo đúng schema:
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
