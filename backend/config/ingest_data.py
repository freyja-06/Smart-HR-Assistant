from extract_cv.RAG.get_data import load_cv_data, load_company_docs_data, load_cv_bm25_index, load_company_docs_bm25_index
from loading_and_caching import load_or_create
import numpy as np
import backend.constant_variables as const


LANGDOCS_SAVE_DIR = const.LANGDOCS_SAVE_DIR

CV_EMBEDDING_SAVE_DIR = const.EMBEDDING_SAVE_DIR
COMPANY_EMBEDDING_SAVE_DIR = const.EMBEDDING_SAVE_DIR 

def main():
    print("="*50)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH NẠP DỮ LIỆU VÀO VECTOR DATABASE")
    print("="*50)
    
    try:
        print("\n[1/2] Đang xử lý thư mục chứa CVs (Trích xuất thông tin và lưu trữ)...")
        # Quá trình này sẽ mất thời gian vì phải đọc PDF và gọi API
        cv_store, cv_embeddings, cv_docs = load_cv_data()
        load_or_create(data = cv_docs, folder_path  = LANGDOCS_SAVE_DIR , var_name = "cv_docs") # Lưu cv docs
        np.save(CV_EMBEDDING_SAVE_DIR, cv_embeddings) # Lưu các vector embedding của cv
        load_cv_bm25_index() # Lưu dữ liệu bm25 index


        print("Hoàn tất nạp dữ liệu CV")
        
        print("\n[2/2] Đang xử lý thư mục chứa company documents (Phân mảnh văn bản và lưu trữ)...")

        company_docs_store, company_embeddings, company_docs = load_company_docs_data()
        load_or_create(data = company_docs, folder_path = LANGDOCS_SAVE_DIR , var_name = "company_docs") # lưu company docs
        np.save(COMPANY_EMBEDDING_SAVE_DIR, company_embeddings)
        load_company_docs_bm25_index()


        print("Hoàn tất nạp dữ liệu company documents")
        print("\nXONG! Tất cả dữ liệu đã được lưu an toàn.")
        
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi trong quá trình nạp dữ liệu: {e}")

if __name__ == "__main__":
    main()