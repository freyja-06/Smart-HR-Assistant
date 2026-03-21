from backend.state.graph_state import GraphState
from backend.agents.rag_agents import sub_query_agent
from backend.agents.coordinator import Task
from langgraph.graph import StateGraph, START, END
from backend.extract_cv.RAG.rag_backend import general_retrieve
from typing import Dict
import asyncio

async def asyn_general_retrieve(subquery, db_type, k, alpha, history_store):
    return await asyncio.to_thread(
        general_retrieve,
        subquery,
        db_type,
        k,
        alpha,
        history_store
    )

rag_workflow = StateGraph(GraphState)

def get_prompt_for_retrieve(task: Task):
    return f"""
        Lệnh truy vấn: {task.instruction} \n
        Các tham số tùy chọn trích xuất từ câu lệnh trên: {task.args} \n
        Nguồn dữ liệu: {task.data_source} \n
        
    """

def optimize_query_node(state: GraphState):
    task_id = state["current_task_id"]
    task = state["plan"].tasks[task_id]

    print(f"Đang chạy task {task_id}: {task.instruction}")

    search_queries = sub_query_agent.invoke(get_prompt_for_retrieve(task))

    return {"search_queries":  search_queries}

async def parallel_retrieve_node(state: GraphState):
    try:
        search_queries = state.get("search_queries").queries
        history_store = state.get("history_cv_store")
        
        retrieve_tasks = {}
        for i, query in enumerate(search_queries, 1):
            retrieve_tasks[f"{i}. {query.data_source}"] = asyn_general_retrieve(
                subquery=query.sub_query,
                db_type=query.data_source,
                k=query.k,
                alpha=query.alpha,
                history_store=history_store
            )

        # Await trực tiếp gather
        results = await asyncio.gather(*retrieve_tasks.values())
        results_dict = dict(zip(retrieve_tasks.keys(), results))

        # Phân loại tài liệu
        cv_docs = []
        company_docs = []
        for key, docs in results_dict.items():
            if "CV_DATABASE" in key or "HISTORY_CV_DATABASE" in key:
                cv_docs.extend(docs)
            else:
                company_docs.extend(docs)

        # Xóa trùng lặp (Logic tiền xử lý dữ liệu của RAG, nên giữ ở Subgraph)
        seen = set()
        unique_cv_docs = []
        for doc in cv_docs:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                unique_cv_docs.append(doc)

        # CHỈ TRẢ VỀ DỮ LIỆU ĐÃ RETRIEVE
        return {
            "cv_documents": unique_cv_docs,
            "company_documents": company_docs,
        }

    except Exception as e:
        # Chỉ trả về error, để parent graph quyết định đánh dấu fail task
        return {
            "module_outputs": {
                "error": str(e)
            }
        }
    

rag_workflow.add_node("optimize_query_node", optimize_query_node)
rag_workflow.add_node("parallel_retrieve_node", parallel_retrieve_node)

rag_workflow.add_edge(START, "optimize_query_node")
rag_workflow.add_edge("optimize_query_node", "parallel_retrieve_node")
rag_workflow.add_edge("parallel_retrieve_node", END)
