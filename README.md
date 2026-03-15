# 🤖 HR Assistant - Intelligent HR Support System

HR Assistant là hệ thống trợ lý nhân sự thông minh được xây dựng trên kiến trúc **Multi-Agent** sử dụng **LangGraph**.  
Hệ thống có khả năng xử lý các tác vụ phức tạp trong quản lý nhân sự như:

- Tra cứu tri thức nội bộ bằng **RAG**
- Soạn thảo **email ứng viên tự động**
- Tạo **bộ câu hỏi phỏng vấn và báo cáo PDF**

Dự án được thiết kế theo mô hình **Orchestrator-Workers**, cho phép mở rộng dễ dàng các module nghiệp vụ.

---

# 🚀 System Architecture

Hệ thống gồm một **Orchestrator Graph** điều phối các **Subgraphs chuyên biệt** để xử lý từng loại tác vụ.

## 1. Orchestrator Graph

Luồng xử lý chính của hệ thống.

Components:

- **Manager Agent**  
  Phân tích truy vấn của người dùng và tạo **Execution Plan**

- **Router Node**  
  Điều hướng request đến module phù hợp

- **Subgraphs**  
  Các module nghiệp vụ độc lập

- **Response Agent**  
  Tổng hợp kết quả từ các module và trả về cho người dùng

---

## 2. Business Modules (Subgraphs)

### 📚 RAG Module

Module xử lý truy xuất tri thức nội bộ.

Pipeline:

1. Query Rewriting
2. Vector Search trên ChromaDB
3. Context Filtering
4. LLM Answer Generation

---

### 📧 Email Module

Tự động soạn thảo email ứng viên.

Tính năng:

- Draft email tự động
- Human-in-the-loop approval
- Người dùng có thể **Accept / Deny** trước khi gửi

---

### 🧑‍💼 Interview Brief Module

Chuẩn bị tài liệu cho vòng phỏng vấn.

Chức năng:

- Thu thập thông tin ứng viên
- Sinh bộ câu hỏi phỏng vấn
- Xuất **Interview Report PDF**

---

# 📂 Project Structure

```
backend/
├── api/                # REST API endpoints (Chat, Session)
├── agents/             # Agent logic (Manager, Response)
├── graphs/             # LangGraph workflows
│   ├── main_graph.py
│   ├── rag_subgraph.py
│   ├── email_subgraph.py
│   └── interview_subgraph.py
├── services/           # External services (ChromaDB, Email, PDF)
├── state/              # GraphState schema and session state
└── main.py             # Application entry point
```

---

# 🛠 Tech Stack

**Framework**

- LangChain
- LangGraph

**LLMs**

- OpenAI GPT-4o
- Claude 3.5 Sonnet

**Backend**

- FastAPI (Python)

**Vector Database**

- ChromaDB

**State Management**

- Session state for conversation memory

---

# 🔄 Processing Pipeline

## 1️⃣ Request Initialization

Frontend gửi request tới backend:

- Load Session State
- Cập nhật Chat History

---

## 2️⃣ Planning Phase

Manager Agent:

- Phân tích user query
- Tạo workflow thực thi

---

## 3️⃣ Execution Phase

Router điều hướng đến module phù hợp.

### RAG Request

```
User Query
 → Query Rewriting
 → Vector Search (ChromaDB)
 → Context Filtering
 → LLM Response
```

### Email Request

```
User Query
 → Generate Email Draft
 → Human Approval (Accept / Deny)
 → Final Output
```

### Interview Preparation

```
Candidate Info
 → Generate Interview Questions
 → Create Interview Brief
 → Export PDF
```

---

## 4️⃣ Response

- Cập nhật GraphState
- Lưu Session
- Trả kết quả về UI

---

# 💻 Installation

## Requirements

- Python **3.10+**
- Docker (recommended for ChromaDB)

---

## 1. Clone Repository

```bash
git clone https://github.com/freyja-06/Smart-HR-Assistant
cd hr-assistant-langgraph
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Environment Configuration

Tạo file `.env`:

```
OPENAI_API_KEY=your_api_key_here
CHROMADB_HOST=localhost
SMTP_SERVER=your_smtp_server
```

---

## 4. Run Application

```bash
python backend/main.py
```

Server sẽ chạy tại:

```
http://localhost:8000
```

---

# 📌 Future Improvements

- Multi-tenant HR knowledge base
- Slack / Teams integration
- Email auto-sending workflow
- Candidate evaluation scoring
- Dashboard analytics

---

# 🤝 Contributing

Pull requests được hoan nghênh.

Nếu phát hiện bug hoặc có đề xuất tính năng mới, vui lòng tạo **Issue**.

---

# 📄 License

Distributed under the **MIT License**.

---

# 👨‍💻 Authors

Developed by **Your Name / Team**
