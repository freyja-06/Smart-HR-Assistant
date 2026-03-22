from backend.state.graph_state import GraphState
from backend.agents.rag_agents import sub_query_agent
from backend.agents.coordinator import Task
from langgraph.graph import StateGraph, START, END
from backend.extract_cv.RAG.rag_backend import general_retrieve
from backend.agents.rag_agents import context_compressor_agent
from typing import Dict
import asyncio


async def asyn_general_retrieve(subquery, db_type, alpha, history_store, k=None):
    return await asyncio.to_thread(
        general_retrieve,
        subquery=subquery,
        db_type=db_type,
        alpha=alpha,
        history_store=history_store,
        k=k
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
                alpha=query.alpha,
                history_store=history_store,
                k=query.k  # Lúc này query.k có thể mang giá trị int hoặc None
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
    
async def context_compressor_node(state: GraphState):
    query = state.get("user_input", "")
    
    # Gộp tất cả tài liệu truy xuất được để nén ngữ cảnh
    all_docs = state.get("company_documents") or []
    
    if not all_docs:
        return {"company_compressed_context": ""}
    compressed_ctx = await context_compressor_agent(
        query=query, 
        docs=all_docs
    )
    
    return {"company_compressed_context": compressed_ctx}

rag_workflow.add_node("optimize_query_node", optimize_query_node)
rag_workflow.add_node("parallel_retrieve_node", parallel_retrieve_node)
rag_workflow.add_node("context_compressor_node", context_compressor_node)

rag_workflow.add_edge(START, "optimize_query_node")
rag_workflow.add_edge("optimize_query_node", "parallel_retrieve_node")
rag_workflow.add_edge("parallel_retrieve_node", "context_compressor_node")
rag_workflow.add_edge("context_compressor_node", END)
