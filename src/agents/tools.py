from typing import List, Callable
from langchain_core.tools import tool
from config.database import cv_store, policies_store

@tool
def send_retriever(query: str, db_type: str, k: int = 4) -> str:
    """Sử dụng công cụ này để tìm kiếm thông tin từ cơ sở dữ liệu.
    
    Args:
        query (str): Câu hỏi, yêu cầu hoặc từ khóa cần tìm kiếm (VD: "Ứng viên biết Python", "Quy định nghỉ phép").
        db_type (str): CHỈ ĐƯỢC CHỌN 1 TRONG 2 GIÁ TRỊ SAU: 
                       - "cv" (Nếu câu hỏi tìm kiếm thông tin ứng viên, kinh nghiệm, kỹ năng)
                       - "policy" (Nếu câu hỏi về quy định, luật, chính sách công ty)
        k (int): Số lượng tài liệu tối đa cần lấy. Mặc định là 4.
    """
    
    if db_type.lower() == "cv":
        retriever = cv_store.as_retriever(search_kwargs={"k": k})
    elif db_type.lower() == "policy":
        retriever = policies_store.as_retriever(search_kwargs={"k": k})
    else:
        return f"Lỗi: Không nhận diện được db_type '{db_type}'. Vui lòng chỉ dùng 'cv' hoặc 'policy'."
        
    # 2. Tool THỰC HIỆN CÔNG VIỆC TÌM KIẾM 
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        return f"Đã xảy ra lỗi khi tìm kiếm: {str(e)}"
    
    # 3. CHUYỂN ĐỔI KẾT QUẢ THÀNH VĂN BẢN (Text) cho LLM đọc
    if not docs:
        return "Hệ thống không tìm thấy bất kỳ thông tin nào liên quan đến yêu cầu này trong cơ sở dữ liệu."
    
    # Gom nội dung các tài liệu lại thành 1 đoạn string lớn
    result_text = f"Tìm thấy {len(docs)} tài liệu liên quan:\n\n"
    for i, d in enumerate(docs, 1):
        # Trích xuất cả nội dung lẫn metadata để LLM biết nguồn gốc tài liệu
        result_text += f"--- Tài liệu {i} ---\n"
        result_text += f"Nội dung: {d.page_content}\n"
        result_text += f"Siêu dữ liệu (Metadata/Nguồn): {d.metadata}\n\n"
        
    # Trả về chuỗi văn bản dài chứa đầy đủ ngữ cảnh để LLM phân tích
    return result_text

tools: List[Callable] = [send_retriever]