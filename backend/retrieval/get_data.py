"""
get_data.py — Legacy module.
==============================
Logic nạp dữ liệu (ingest) đã được chuyển sang:
  - backend/data_ingestion/pipeline.py  (orchestrator)
  - backend/data_ingestion/storage.py   (unified storage)

Logic truy xuất dữ liệu (runtime) nằm tại:
  - backend/config/database.py          (load các artifact đã lưu)
  - backend/retrieval/rag_backend.py    (retrieval pipeline)

File này được giữ lại nhưng không còn logic.
Nếu không có module nào import từ đây, có thể xóa an toàn.
"""
