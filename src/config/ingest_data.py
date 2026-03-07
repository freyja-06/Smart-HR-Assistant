import sys
import os

# Đường dẫn hiện tại: src/config
current_dir = os.path.dirname(os.path.abspath(__file__))

# Lùi lên một cấp để lấy thư mục src
src_dir = os.path.dirname(current_dir)

# Thêm thư mục src vào sys.path
sys.path.append(src_dir)


from extract_cv.RAG.rag_backend import load_cv_data, load_policies_data

def main():
    print("="*50)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH NẠP DỮ LIỆU VÀO VECTOR DATABASE")
    print("="*50)
    
    try:
        print("\n[1/2] Đang xử lý thư mục chứa CVs (Trích xuất thông tin qua Gemini và lưu trữ)...")
        # Quá trình này sẽ mất thời gian vì phải đọc PDF và gọi API
        cv_store = load_cv_data()
        print("✅ Hoàn tất nạp dữ liệu CVs vào collection 'CVs'.")
        
        print("\n[2/2] Đang xử lý thư mục chứa Policies (Phân mảnh văn bản và lưu trữ)...")
        policies_store = load_policies_data()
        print("✅ Hoàn tất nạp dữ liệu Policies vào collection 'Policies'.")
        
        print("\n🎉 XONG! Tất cả dữ liệu đã được lưu an toàn vào thư mục 'chroma_db'.")
        print("Bây giờ bạn có thể chạy Agent một cách cực kỳ nhanh chóng.")
        
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi trong quá trình nạp dữ liệu: {e}")

if __name__ == "__main__":
    main()