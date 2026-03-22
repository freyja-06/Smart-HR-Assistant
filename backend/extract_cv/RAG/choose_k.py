import math
import numpy as np

def compute_base_k(n_docs: int):

    """
    Hàm lựa chọn k theo dataset
    Chọn 

        k ~ sqrt(N) sẽ cho balance tốt giữa recall và precision
    
        Lower bound = 10 để tránh case khi N nhỏ → sqrt(N) < 5
        dẫn đến không đủ context cho LLM

        Upper bound = 100 để tránh rerank quá chậm (vì cross-encoder rất đắt)
        thực tế thì rerank >100 docs thường không improve nhiều

    """

    return int(min(100, max(10, math.sqrt(n_docs))))


def adjust_k_by_vector_scores(base_k, vector_scores_raw):

    """
    Hàm điều chỉnh k theo độ phân tán (độ lệch chuẩn) của vector - cũng có thể coi là độ khó của query: 
    
    Logic chính của hàm : Ta sẽ có 2 loại trường hợp quan trọng:
        1. Nếu độ lệch chuẩn lớn - tức là độ tương hợp với query phân tán mạnh trên từng document

            Ví dụ với trường hợp này:
                [0.95, 0.80, 0.40, 0.20, 0.10]

            Có thể thấy có vài doc rất relevant (index 0, 1, 2) \n
            phần còn lại rõ ràng kém hơn (index 3,4) \n
            Query “dễ”, model phân biệt tốt \n
            => Ta giữ k nhỏ

        2. Nếu độ lệch chuẩn thấp - ngược lại với trường hợp trên

            Ví dụ như: 
                [0.52, 0.50, 0.49, 0.48, 0.47]

            Dễ thấy rằng tất cả doc “na ná nhau”, khiến model không chắc cái nào đúng \n
            Query “khó / ambiguous”
            => Ta giữ k lớn để tăng tính tổng quát và đô chính xác
    """

    if len(vector_scores_raw) < 2:
        return base_k

    std = np.std(vector_scores_raw)

    # normalize std về [0,1]
    norm_std = std / (np.mean(vector_scores_raw) + 1e-8)

    # std thấp → scores gần nhau → query khó → tăng k
    factor = 1 + (1 - norm_std)

    return int(base_k * factor)


def adjust_k_by_bm25(base_k, bm25_scores):

    """
    Hàm điều chỉnh k dựa trên phân phối điểm số của BM25 (đại diện cho keyword matching - tìm kiếm từ khóa).
    
    Logic chính: Ta xem xét tỷ lệ (ratio) giữa điểm cao nhất (top) và điểm trung bình (mean) của các document.
    
        1. Tỷ lệ cao (ratio > 3): 
            Điều này có nghĩa là có một (hoặc vài) document chứa từ khóa cực kỳ sát với query, 
            điểm số vượt trội hoàn toàn so với mức trung bình của phần còn lại.
            => Model bắt đúng "key" rất tốt, document rất "dính". 
            => Ta có thể tự tin giảm k xuống (nhân với 0.7) để tăng tốc độ xử lý mà không sợ rớt mất context quan trọng.
            
        2. Tỷ lệ thấp (ratio < 1.5):
            Điểm số dàn trải đều đều, đỉnh (top) không cách biệt mấy so với mức trung bình.
            Điều này thường xảy ra khi câu hỏi chứa các từ vựng phổ biến (stop words) hoặc không khớp chính xác keyword đặc thù nào.
            => Query này đang bị "mờ" về mặt từ khóa (Fuzzy). 
            => Ta cần tăng k lên (nhân với 1.3) để "vớt" thêm tài liệu, cho LLM hoặc Cross-Encoder tự đánh giá chéo về sau.
    """

    if len(bm25_scores) < 2:
        return base_k

    top = np.max(bm25_scores)
    mean = np.mean(bm25_scores)

    # peak ratio
    ratio = top / (mean + 1e-8)

    # ratio cao → match tốt → giảm k
    if ratio > 3:
        return int(base_k * 0.7)
    elif ratio < 1.5:
        return int(base_k * 1.3)

    return base_k


def adaptive_k(
    n_docs: int,
    vector_scores_raw: np.ndarray,
    bm25_scores_full: np.ndarray,
    k_min: int = 10,
    k_max: int = 100
):
    
    """
    Hàm tổng hợp (Orchestrator) để tính toán ra giá trị k động (adaptive k) cuối cùng.
    Hàm này kết hợp đánh giá độ khó của query trên cả 2 phương diện: ngữ nghĩa (Vector) và từ khóa (BM25).
    
    Các bước thực hiện:
        1. Tính base_k: Số lượng k nền tảng phụ thuộc vào quy mô (size) của cơ sở dữ liệu.
        2. Filter qua Vector: Lấy base_k đi qua bộ lọc Vector để điều chỉnh dựa trên độ phân tán ngữ nghĩa.
        3. Filter qua BM25: Lấy base_k đi qua bộ lọc BM25 để điều chỉnh dựa trên độ nổi bật của từ khóa.
        4. Hybrid Average: Lấy trung bình cộng của k_vec và k_bm25 để hệ thống cân bằng được sức mạnh của cả hai phương pháp (Hybrid Search).
        5. Clamping: Cuối cùng, dùng hàm max/min để "chốt chặn" giá trị k, đảm bảo nó luôn nằm trong khoảng an toàn đã định [k_min, k_max] để tránh lỗi thiếu hụt hoặc tràn RAM.
    """

    base_k = compute_base_k(n_docs)

    k_vec = adjust_k_by_vector_scores(base_k, vector_scores_raw)
    k_bm25 = adjust_k_by_bm25(base_k, bm25_scores_full)

    k_final = int((k_vec + k_bm25) / 2)

    return max(k_min, min(k_final, k_max))