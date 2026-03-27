from backend.agents.coordinator import Plan
from backend.agents.rag_agents import ListSubQuery
from typing import TypedDict, List, Optional, Dict, Any, Annotated, Literal
from langchain_core.documents import Document
from langgraph.graph import StateGraph
"""
-----------------------State Update Protocols----------------------
1. Xác định chính xác tác vụ đang làm:

    Đọc current_task_id từ State, sau đó tìm trong plan để 
    lấy đúng instruction và args 
    phục vụ cho việc thực thi.

2. Đánh dấu hoàn thành (Thành công):

    Sau khi chạy xong logic của node (truy vấn ChromaDB, viết email,...), 
    node phải lấy danh sách completed_tasks hiện tại, thêm current_task_id vào đó, 
    và trả về để ghi đè lên State. 
    Nếu không làm vậy, router_node sẽ đọc lại plan
    và tiếp tục bắt chạy lại chính task vừa làm.

3. Đánh dấu lỗi (Thất bại):

    Mọi logic thực thi nên được bọc trong try...except. 
    Nếu có lỗi xảy ra (ví dụ: API LLM sập, lỗi kết nối DB), 
    node phải ném current_task_id vào failed_tasks. 
    Nhờ vậy, router_node sẽ biết tác vụ này hỏng và bỏ qua nó để đi tới tác vụ tiếp theo 
    (hoặc về GENERAL_CHAT).

Sample: một node xử lý tác vụ

    def rag_search_node(state: GraphState):
        task_id = state["current_task_id"]
        task = next(t for t in state["plan"].tasks if t.task_id == task_id)

        print(f"Đang chạy task {task_id}: {task.instruction}")

        # xử lý logic...

        return {
            "completed_tasks": state.get("completed_tasks", []) + [task_id]
        }


"""

def print_state(state: dict, node_name: str = "CURRENT NODE"):
    """
    Hàm in ra trạng thái của GraphState một cách đẹp mắt và dễ nhìn để debug.
    
    Args:
        state: GraphState hiện tại
        node_name: Tên của node đang gọi hàm này (để dễ tracking)
    """
    print("\n" + "═"*65)
    print(f"📊 [DEBUG-STATE] TRẠNG THÁI TẠI: {node_name.upper()}")
    print("═"*65)
    
    # 1. Thông tin điều hướng cơ bản
    print(f"🟢 User Input      : {state.get('user_input')}")
    print(f"🔄 Current Route   : {state.get('current_route', 'Chưa xác định')}")
    print(f"🆔 Current Task ID : {state.get('current_task_id', 'None')}")
    print("-" * 65)
    
    # 2. Kế hoạch và Tiến độ (Plan & Tasks)
    completed = state.get('completed_tasks', [])
    failed = state.get('failed_tasks', [])
    plan = state.get('plan')
    
    if plan and hasattr(plan, 'tasks') and plan.tasks:
        print(f"📋 Plan Tasks      : {len(plan.tasks)} task(s) (✅ {len(completed)} xong | ❌ {len(failed)} lỗi)")
        for t in sorted(plan.tasks, key=lambda x: x.task_id):
            # Cắt ngắn instruction nếu quá dài
            instr_preview = t.instruction[:45] + "..." if len(t.instruction) > 45 else t.instruction
            
            # Xác định icon trạng thái
            if t.task_id in completed:
                status_icon = "✅"
            elif t.task_id in failed:
                status_icon = "❌"
            else:
                status_icon = "⏳"
                
            print(f"   ├─ [{status_icon} ID: {t.task_id}] {t.route}")
            print(f"   │  └─ Hướng dẫn: {instr_preview}")
    else:
        print("📋 Plan            : Chưa có kế hoạch (None/Rỗng)")
        
    print("-" * 65)
        
    # 3. Dữ liệu truy xuất (RAG & Context)
    cv_docs = state.get('cv_documents') or []
    company_docs = state.get('company_documents') or []
    print(f"📄 CV Docs         : {len(cv_docs)} tài liệu")
    print(f"🏢 Company Docs    : {len(company_docs)} tài liệu")
    
    # Cắt ngắn context nén
    compressed_ctx = state.get('company_compressed_context')
    if compressed_ctx:
        ctx_preview = compressed_ctx.replace('\n', ' ')[:50] + "..." if len(compressed_ctx) > 50 else compressed_ctx
        print(f"🗜️ Compressed Ctx  : {ctx_preview}")
    else:
        print(f"🗜️ Compressed Ctx  : Rỗng")
        
    # Kiểm tra bộ nhớ history
    history_store = state.get('history_cv_store')
    if history_store and isinstance(history_store, dict):
        print(f"🗄️ History Store   : {len(history_store.get('docs', []))} bản ghi đang lưu")
    else:
        print(f"🗄️ History Store   : Rỗng")

    print("-" * 65)

    # 4. Kết quả đầu ra của các module khác
    print(f"🛠️ Module Outputs  : {state.get('module_outputs', {})}")
    
    # Chỉ in các module này nếu chúng có dữ liệu để tránh rối mắt
    if state.get('email_draft'): 
        print("📧 Email Draft     : [Đã tạo bản nháp]")
    if state.get('email_sent'): 
        print("📤 Email Sent      : [Thành công]")
    if state.get('interview_brief'): 
        print("📝 Interview Brief : [Đã tạo]")
    if state.get('interview_pdf_path'): 
        print(f"📑 Interview Path  : {state.get('interview_pdf_path')}")
        
    print("═"*65 + "\n")

def get_next_task(plan, completed, failed):
    for task in sorted(plan.tasks, key=lambda t: t.task_id):
        task_id = task.task_id

        if task_id in completed or task_id in failed:
            continue

        # check dependencies
        if all(dep in completed for dep in task.dependencies):
            return task

    return None

class GraphState(TypedDict):
    """
    Global shared state across the entire LangGraph workflow.

    Each field represents a piece of data produced or consumed
    by different nodes in the pipeline.
    """

    # =========================================================
    # USER INPUT
    # =========================================================

    user_input: Annotated[
        str,
        "Original request from the HR user that starts the workflow."
    ]


    # =========================================================
    # EXECUTION CONTROL
    # =========================================================

    plan: Annotated[
        Optional["Plan"],
        "Execution plan generated by the Manager Agent containing ordered tasks."
    ]

    current_task_id: Annotated[
        Optional[int],
        "ID of the task currently being executed in the plan."
    ]

    current_route: Annotated[
        Literal["RAG_SEARCH", "WRITE_EMAIL", "GENERATE_INTERVIEW_BRIEF" , "GENERAL_CHAT"],
        "Route name indicating which module should execute the current task."
    ]

    completed_tasks: Annotated[
        Optional[List[int]],
        "List of task IDs that have successfully completed execution."
    ]

    failed_tasks: Annotated[
        Optional[List[int]],
        "List of task IDs that failed during execution."
    ]

    # =========================================================
    # RAG PIPELINE DATA
    # =========================================================

    search_queries: Annotated[
        Optional[ListSubQuery],
        "List query rewritten by the LLM to optimize retrieval from the vector database."
    ]

    cv_documents: Annotated[
        Optional[List[Document]],
        "CVs documents retrieved."
    ]

    history_cv_store: Annotated[
        Optional[Dict],
        "CVs documents retrieved from the previous query."
    ]

    company_documents: Annotated[
        Optional[List[Document]],
        "Company documents retrieved."
    ]

    company_compressed_context: Annotated[
        Optional[str],
        "Ngữ cảnh đã được nén bằng Map-Reduce từ các tài liệu truy xuất được. Chỉ sử dụng cho company documents"
    ]

    # =========================================================
    # MODULE OUTPUTS
    # =========================================================

    # ----- Email Module -----

    email_draft: Annotated[
        Optional[str],
        "Draft email generated by the email module before user confirmation."
    ]

    email_sent: Annotated[
        Optional[bool],
        "Indicates whether the email was successfully sent."
    ]


    # ----- Interview Brief Module -----

    interview_brief: Annotated[
        Optional[str],
        "Generated structured interview preparation content."
    ]

    interview_pdf_path: Annotated[
        Optional[str],
        "Filesystem path to the generated interview brief PDF."
    ]


    # ----- Generic Outputs -----

    module_outputs: Annotated[
        Optional[Dict[str, Any]],
        "Flexible storage for outputs from modules that do not have dedicated fields."
    ]


    # =========================================================
    # FINAL RESPONSE
    # =========================================================

    final_answer: Annotated[
        Optional[str],
        "Final response synthesized by the Response Agent and returned to the user."
    ]

workflow = StateGraph(GraphState)