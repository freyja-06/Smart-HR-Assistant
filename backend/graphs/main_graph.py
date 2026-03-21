from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver
from backend.agents.coordinator import manager_agent, response_agent
from backend.state.graph_state import GraphState, workflow
from backend.graphs.rag_subgraph import rag_workflow
from backend.extract_cv.RAG.rag_backend import build_history_store, update_history_store
import backend.constant_variables as const
from backend.agents.llm_processor.llm_factory import ModelFactory

memory = InMemorySaver()
# Sau này sẽ sử dụng Dùng SqliteSaver hoặc DB tương tự

rag_app = rag_workflow.compile()


def manager_node(state: GraphState):
    print("\n-> Manager agent đang phân chia công việc!")
    response = manager_agent.invoke({"user_input": state["user_input"]})
    return {"plan": response}

def router_node(state: GraphState):
    plan = state.get("plan")
    completed = state.get("completed_tasks", [])
    failed = state.get("failed_tasks", [])

    if not plan or not plan.tasks:
        return {
            "current_route": "GENERAL_CHAT",
            "current_task_id": None
        }

    # tìm task tiếp theo hợp lệ
    for task_id in sorted(plan.tasks.keys()):
        task = plan.tasks[task_id]

        if task_id in completed or task_id in failed:
            continue

        if all(dep in completed for dep in task.dependencies):
            return {
                "current_route": task.route,
                "current_task_id": task_id
            }

    return {
        "current_route": "GENERAL_CHAT",
        "current_task_id": None
    }


def route_condition(state: GraphState) -> str:
    """Hàm này đọc state và trả về TÊN NODE tiếp theo cần chạy"""
    route = state.get("current_route")
    
    if route == "RAG_SEARCH":
        return "rag_search_node"
    
    elif route == "WRITE_EMAIL":
        return "write_email_node"
    
    elif route == "GENERATE_INTERVIEW_BRIEF":
        return "generate_interview_brief_node"

    else:
        return "general_chat_node"


def general_chat_node(state: GraphState):
    print("Agent đang trả về kết quả cuói cùng!")
    response = response_agent.invoke(state)
    return {"final_answer": response}

def rag_search_node(state: GraphState):
    task_id = state["current_task_id"]
    
    # 1. Gọi Subgraph để lấy dữ liệu (LangGraph tự tương thích async/sync ở đây)
    result = rag_app.invoke(state)

    # 2. Xử lý logic Thất bại (nếu subgraph có lỗi)
    if result.get("module_outputs", {}).get("error"):
        return {
            "failed_tasks": state.get("failed_tasks", []) + [task_id],
            "module_outputs": result["module_outputs"]
        }

    # 3. Lấy dữ liệu trả về từ Subgraph
    unique_cv_docs = result.get("cv_documents", [])
    company_docs = result.get("company_documents", [])

    # 4. Xử lý Business Logic: Cập nhật History Store
    old_history = state.get("history_cv_store")

    if old_history is None:
        new_history_store = build_history_store(
            docs=unique_cv_docs,
            embedding_model = ModelFactory.create(
                model_type="embedding",
                provider="ollama",
                model_name="nomic-embed-text",
            )
        )
    else:
        new_history_store = update_history_store(
            history_store=old_history,
            new_docs=unique_cv_docs,
            embedding_model = ModelFactory.create(
                model_type="embedding",
                provider="ollama",
                model_name="nomic-embed-text",
            )
        )

    # 5. Xử lý Business Logic: Đánh dấu hoàn thành Task
    return {
        "cv_documents": unique_cv_docs,
        "company_documents": company_docs,
        "history_cv_store": new_history_store,
        "completed_tasks": state.get("completed_tasks", []) + [task_id]
    }


def write_email_node(state: GraphState):
    pass # sẽ viết sau

def generate_interview_brief_node(state: GraphState):
    pass # sẽ viết sau

#----------------------------------------------------

workflow.add_node("manager_node", manager_node)
workflow.add_node("router_node", router_node)

workflow.add_node("rag_search_node", rag_search_node)
workflow.add_node("write_email_node", write_email_node)
workflow.add_node("generate_interview_brief_node", generate_interview_brief_node)
workflow.add_node("general_chat_node", general_chat_node)


workflow.add_edge(START, "manager_node")
workflow.add_edge("manager_node", "router_node")
workflow.add_conditional_edges("router_node", route_condition, 
    {
        "rag_search_node":                "rag_search_node",
        "write_email_node":               "write_email_node",
        "generate_interview_brief_node":  "generate_interview_brief_node",
        "general_chat_node":              "general_chat_node"
    }
)

workflow.add_edge("rag_search_node", "router_node")
workflow.add_edge("write_email_node", "router_node")
workflow.add_edge("generate_interview_brief_node", "router_node")

workflow.add_edge("general_chat_node", END)


# Biên dịch và chạy dồ thị

app = workflow.compile(checkpointer= memory)

def run(state: GraphState, config: dict):
    result = app.invoke(state, config=config)
    return result